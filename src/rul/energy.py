"""Operation counting + 45 nm energy model (Horowitz, ISSCC 2014 numbers as used in SNN literature).

E_MAC = 4.6 pJ (32-bit float multiply 3.7 + add 0.9)   E_AC = 0.9 pJ (32-bit float add)
ARITHMETIC ONLY: memory traffic is ignored (it usually dominates in real silicon) -> a ceiling on gains.
"""
import math
from typing import Dict

import torch
import torch.nn as nn

E_MAC_PJ = 4.6
E_AC_PJ = 0.9


def count_macs(model: nn.Module, T: int = 30, n_features: int = 14) -> int:
    """Dense multiply-accumulates for ONE window, via forward hooks (+ analytic rules for recurrences)."""
    total = [0]
    hooks = []

    def lin(m, i, o):
        total[0] += i[0].numel() // m.in_features * m.in_features * m.out_features

    def conv(m, i, o):
        total[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0]

    def rnn(m, i, o):
        x = i[0]
        L = x.size(1) if m.batch_first else x.size(0)
        gates = {"LSTM": 4, "GRU": 3, "RNN": 1}[m.__class__.__name__]
        d = 2 if m.bidirectional else 1
        in_sz = m.input_size
        for _ in range(m.num_layers):
            total[0] += d * L * gates * (in_sz * m.hidden_size + m.hidden_size ** 2)
            in_sz = m.hidden_size * d

    def mha(m, i, o):
        L, E = i[0].size(1), m.embed_dim
        total[0] += 2 * L * L * E  # QK^T and AV (the q/k/v/out projections are Linear -> counted there)

    def s6(m, i, o):
        L = i[0].size(1)
        total[0] += 4 * L * m.d_inner * m.d_state  # dt*A, dA*h, dB*u, h.C per step

    for mod in model.modules():
        n = mod.__class__.__name__
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin))
        elif isinstance(mod, nn.Conv1d):
            hooks.append(mod.register_forward_hook(conv))
        elif n in ("LSTM", "GRU", "RNN"):
            hooks.append(mod.register_forward_hook(rnn))
        elif isinstance(mod, nn.MultiheadAttention):
            hooks.append(mod.register_forward_hook(mha))
        elif n == "S6Block":
            hooks.append(mod.register_forward_hook(s6))
    was_training = model.training
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)  # fast path bypasses submodule hooks
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, T, n_features, device=dev))
    torch.backends.mha.set_fastpath_enabled(fast)
    model.train(was_training)
    for h in hooks:
        h.remove()
    return int(total[0])


def energy_pj(macs: float, acs: float = 0.0) -> float:
    return macs * E_MAC_PJ + acs * E_AC_PJ
