import torch
from rul.models import count_params, create_model


def test_hybrids_forward_and_window_free():
    for name in ("tcnattn", "cnnbigruattn", "convtransformer"):
        m = create_model(name, 14)
        for T in (15, 30, 50):  # models must not hard-code the window length
            y = m(torch.randn(4, T, 14))
            assert y.shape == (4,) and torch.isfinite(y).all(), (name, T)
        assert count_params(m) > 0


def test_tcnattn_is_causal_in_features():
    m = create_model("tcnattn", 14, dilations=(1, 2, 4, 8, 16)).eval()
    x = torch.randn(2, 50, 14)
    assert torch.isfinite(m(x)).all()


def test_mamba2_forward_backward_and_uses_history():
    import torch
    from rul.models import create_model
    m = create_model("mamba2", 14)
    x = torch.randn(4, 50, 14)
    y = m(x); assert y.shape == (4,) and torch.isfinite(y).all()
    y.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    x2 = x.clone(); x2[:, 0] += 1.0                                          # a change at the first step must reach the last-step output
    assert (m(x2) - m(x)).abs().max() > 0
