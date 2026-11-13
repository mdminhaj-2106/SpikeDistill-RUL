import numpy as np
import torch

from rul.distill import DistillationLoss, fit_distillation
from rul.energy import count_macs
from rul.models import AttentionBiLSTMTeacher, SpikeDistillStudent, create_model


def test_loss_terms_finite_and_switchable():
    B, T = 4, 30
    crit = DistillationLoss(teacher_dim=64, student_dim=32)
    attn = torch.softmax(torch.randn(B, T), -1)
    kw = dict(s_rul=torch.rand(B), y=torch.rand(B), t_rul=torch.rand(B), s_v=torch.randn(B, T, 32),
              t_h=torch.randn(B, T, 64), s_attn=attn, t_attn=attn, rate=torch.tensor(0.4))
    full, parts = crit(**kw)
    assert torch.isfinite(full) and parts["sparse"] > 0
    crit.w.update(kd=0, traj=0, attn=0, sparse=0)
    task_only, _ = crit(**kw)
    assert task_only <= full


def test_fit_distillation_smoke_runs_for_spiking_and_plain_students():
    rng = np.random.default_rng(0)
    d = {"X_train": rng.random((64, 30, 5), dtype=np.float32), "y_train": rng.random(64).astype("f") * 125,
         "X_val": rng.random((16, 30, 5), dtype=np.float32), "y_val": rng.random(16).astype("f") * 125,
         "X_test": rng.random((8, 30, 5), dtype=np.float32), "y_test": rng.random(8).astype("f") * 125}
    teacher = AttentionBiLSTMTeacher(5, hidden=8, layers=1)
    for st in (SpikeDistillStudent(5, hidden=8), create_model("smallgru", 5)):
        r = fit_distillation(st, teacher, d, epochs=2, batch=32, verbose=False, device="cpu")
        assert np.isfinite(r["test"]["RMSE"])


def test_mac_counter_matches_hand_count_for_lstm():
    m = create_model("lstm", 14, hidden=64)
    expected = 30 * 4 * (14 * 64 + 64 * 64) + 64 * 32 + 32 * 1
    assert count_macs(m) == expected


def test_bilstm_readout_uses_full_backward_pass():
    # backward half must depend on the FIRST cycles: changing x[:,0] changes the output
    m = create_model("bilstm", 4, hidden=8).eval()
    x = torch.randn(2, 30, 4)
    y0 = m(x)
    x2 = x.clone(); x2[:, 0] += 1.0
    assert not torch.allclose(m(x2), y0)
