"""Unit tests for metrics and all model architectures."""
import numpy as np
import pytest
import torch

from rul.metrics import all_metrics, cmapss_score
from rul.models import (
    AttentionBiLSTMTeacher,
    CNNLSTMRegressor,
    DeepNN,
    MambaRegressor,
    RecurrentRegressor,
    SpikeDistillStudent,
    VanillaKDSmallLSTM,
    WideNN,
    count_params,
    create_model,
)


def test_perfect_prediction_scores_zero():
    y = np.array([10.0, 50.0, 100.0])
    m = all_metrics(y, y)
    assert m["RMSE"] == 0 and m["Score"] == 0 and m["R2"] == 1


def test_late_guess_punished_more_than_early():
    late = cmapss_score([50], [60])   # guessed 10 cycles too late
    early = cmapss_score([50], [40])  # guessed 10 cycles too early
    assert late > early
    assert np.isclose(late, np.exp(1) - 1)
    assert np.isclose(early, np.exp(10 / 13) - 1)


@pytest.mark.parametrize("kind", ["rnn", "gru", "lstm", "bilstm"])
def test_recurrent_baselines(kind):
    x = torch.randn(4, 30, 14)
    model = RecurrentRegressor(n_sensors=14, kind=kind).eval()
    out = model(x)
    assert out.shape == (4,)
    assert not torch.isnan(out).any()


def test_feedforward_baselines():
    x = torch.randn(4, 30, 14)
    wide = WideNN(n_sensors=14, window=30).eval()
    deep = DeepNN(n_sensors=14, window=30).eval()
    assert wide(x).shape == (4,)
    assert deep(x).shape == (4,)
    assert count_params(wide) == 14 * 30 + 1


def test_cnn_lstm():
    x = torch.randn(4, 30, 14)
    model = CNNLSTMRegressor(n_sensors=14).eval()
    out = model(x)
    assert out.shape == (4,)


def test_mamba_s6():
    x = torch.randn(4, 30, 14)
    model = MambaRegressor(n_sensors=14, d_model=32, n_layers=2, d_state=8).eval()
    out = model(x)
    assert out.shape == (4,)
    assert not torch.isnan(out).any()


def test_teacher_attention_bilstm_internals():
    x = torch.randn(4, 30, 14)
    teacher = AttentionBiLSTMTeacher(n_sensors=14, hidden=32, layers=2).eval()
    rul, h_traj, attn = teacher.forward_with_internals(x)
    assert rul.shape == (4,)
    assert h_traj.shape == (4, 30, 64)
    assert attn.shape == (4, 30)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_student_spiking_surrogate_and_internals():
    x = torch.randn(4, 30, 14)
    student = SpikeDistillStudent(n_sensors=14, hidden=32)
    student.train()
    rul, v_traj, s_traj, attn, spike_rate = student.forward_with_internals(x)
    assert rul.shape == (4,)
    assert v_traj.shape == (4, 30, 32)
    assert s_traj.shape == (4, 30, 32)
    assert attn.shape == (4, 30)
    assert 0.0 <= spike_rate.item() <= 1.0

    # Test backward pass with surrogate gradients
    loss = rul.sum() + spike_rate
    loss.backward()
    assert student.lif1.theta_beta.grad is not None
    assert student.lif1.theta_v.grad is not None


def test_model_factory():
    for name in ["widenn", "deepnn", "rnn", "gru", "lstm", "bilstm", "cnnlstm", "mamba", "teacher", "student", "vanillakd"]:
        m = create_model(name, n_sensors=14)
        assert count_params(m) > 0
