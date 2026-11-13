"""Baseline, State-Space (Mamba), Teacher, and Spiking (SNN) Models for RUL Prediction.

All models adhere to the standard contract:
    Input  x: (batch, 30 cycles, n_sensors)
    Output y: (batch,) scalar remaining useful life cycles
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Linear & Feedforward Baselines (WideNN & DeepNN)
# ==============================================================================
class WideNN(nn.Module):
    """Linear regression baseline over flattened temporal window (30 * n_sensors)."""

    def __init__(self, n_sensors: int = 14, window: int = 30):
        super().__init__()
        self.in_features = n_sensors * window
        self.linear = nn.Linear(self.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_sensors)
        x_flat = x.reshape(x.size(0), -1)
        return self.linear(x_flat).squeeze(-1)


class DeepNN(nn.Module):
    """Multi-layer perceptron baseline over flattened temporal window."""

    def __init__(self, n_sensors: int = 14, window: int = 30, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        in_dim = n_sensors * window
        layers = []
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(x.size(0), -1)
        return self.net(x_flat).squeeze(-1)


# ==============================================================================
# 2. Recurrent Baselines (RNN, GRU, LSTM, BiLSTM)
# ==============================================================================
class RecurrentRegressor(nn.Module):
    """Recurrent baseline supporting RNN, GRU, LSTM, and Bidirectional LSTM."""

    CELLS = {
        "rnn": nn.RNN,
        "gru": nn.GRU,
        "lstm": nn.LSTM,
        "bilstm": nn.LSTM,
    }

    def __init__(
        self,
        n_sensors: int = 14,
        kind: str = "lstm",
        hidden: int = 64,
        layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.kind = kind.lower()
        self.is_bidirectional = self.kind == "bilstm"
        cell_cls = self.CELLS[self.kind]

        self.rnn = cell_cls(
            input_size=n_sensors,
            hidden_size=hidden,
            num_layers=layers,
            bidirectional=self.is_bidirectional,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        out_dim = hidden * 2 if self.is_bidirectional else hidden
        self.head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h_n holds the FINAL state of each direction. For a BiLSTM, outputs[:, -1] would give the
        # backward half after seeing only ONE step (a classic bug), so we read h_n instead.
        _, h_n = self.rnn(x)
        h_n = h_n[0] if self.kind in ("lstm", "bilstm") else h_n
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.is_bidirectional else h_n[-1]
        return self.head(h_last).squeeze(-1)


# ==============================================================================
# 3. Hybrid Spatial-Temporal Baseline (CNN-LSTM)
# ==============================================================================
class CNNLSTMRegressor(nn.Module):
    """1D Temporal Convolution followed by an LSTM."""

    def __init__(
        self,
        n_sensors: int = 14,
        conv_channels: int = 32,
        kernel_size: int = 3,
        lstm_hidden: int = 48,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=n_sensors,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, sensors) -> permute to (batch, sensors, time) for Conv1d
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        outputs, _ = self.lstm(x_conv)
        return self.head(outputs[:, -1]).squeeze(-1)


# ==============================================================================
# 4. Pure PyTorch Selective State Space Model (Mamba S6)
# ==============================================================================
class S6Block(nn.Module):
    """Single Selective State Space (S6) block with data-dependent B, C, Delta."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        # Input projection into [u, z]
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Causal Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # Selective projection for Delta, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4 parameterization of A: log(-A) with HiPPO-like initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # 1D causal convolution on u
        u_conv = self.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
        u_conv = F.silu(u_conv)

        # Selective parameters from u_conv
        x_dbl = self.x_proj(u_conv)  # (B, L, dt_rank + 2*d_state)
        dt, B_t, C_t = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
        A = -torch.exp(self.A_log)          # (d_inner, d_state)

        # Discretize and recurrent selective scan
        # h_t = exp(dt * A) * h_{t-1} + (dt * B_t) * u_t
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            B_curr = B_t[:, t, :].unsqueeze(1) # (B, 1, d_state)
            u_curr = u_conv[:, t, :].unsqueeze(-1) # (B, d_inner, 1)
            C_curr = C_t[:, t, :].unsqueeze(-1) # (B, d_state, 1)

            dA = torch.exp(dt_t * A.unsqueeze(0)) # (B, d_inner, d_state)
            dB = dt_t * B_curr                    # (B, d_inner, d_state)
            h = dA * h + dB * u_curr

            y_t = torch.matmul(h, C_curr).squeeze(-1) # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1) + u_conv * self.D  # (B, L, d_inner)
        y = y * F.silu(z)                             # Gated branch
        return self.out_proj(y)                       # (B, L, d_model)


class MambaRegressor(nn.Module):
    """Mamba (Selective State-Space Model) Regressor for Turbofan RUL."""

    def __init__(
        self,
        n_sensors: int = 14,
        d_model: int = 48,
        n_layers: int = 2,
        d_state: int = 16,
    ):
        super().__init__()
        self.embedding = nn.Linear(n_sensors, d_model)
        self.blocks = nn.ModuleList([
            S6Block(d_model=d_model, d_state=d_state) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.norm(h)
        return self.head(h[:, -1]).squeeze(-1)


class SSDBlock(nn.Module):
    """Mamba-2 style block (Dao & Gu, 2024): a multi-head selective SSM whose state decay is ONE scalar per head.

    Mamba-1 decays every (channel, state) pair with its own A. Mamba-2 ties A to a scalar per head and shares B, C across
    heads, which is what lets the recurrence be rewritten as structured attention ("state-space duality", SSD) and run as
    matrix multiplies. For our 30-100 step windows the sequential scan below is exact and cheap; it computes the same
    function as the chunked SSD form.
        h_t = exp(dt_t * a_head) * h_{t-1} + dt_t * (x_t outer B_t)        h_t: (heads, headdim, d_state)
        y_t = h_t C_t + D_head * x_t
    Block: in_proj -> [z | x B C | dt]; depthwise causal conv + SiLU on [x B C]; scan; gated RMSNorm (y * SiLU(z)); out_proj.
    """

    def __init__(self, d_model: int, d_state: int = 16, headdim: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = d_model * expand
        assert self.d_inner % headdim == 0
        self.headdim, self.d_state = headdim, d_state
        self.nheads = self.d_inner // headdim
        self.conv_dim = self.d_inner + 2 * d_state
        self.in_proj = nn.Linear(d_model, self.d_inner + self.conv_dim + self.nheads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, d_conv, padding=d_conv - 1, groups=self.conv_dim)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))           # inverse softplus of dt in [1e-3, 1e-1]
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1.0, 16.0)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm_w = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, L, _ = u.shape
        z, xBC, dt = torch.split(self.in_proj(u), [self.d_inner, self.conv_dim, self.nheads], dim=-1)
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[:, :, :L].transpose(1, 2))
        x, Bm, Cm = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias)                                       # (B, L, H)
        A = -torch.exp(self.A_log)                                               # (H,)
        x = x.reshape(B, L, self.nheads, self.headdim)
        h = x.new_zeros(B, self.nheads, self.headdim, self.d_state)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t] * A)[:, :, None, None]                       # (B, H, 1, 1)
            dBx = (dt[:, t, :, None] * x[:, t])[..., None] * Bm[:, t, None, None, :]   # (B, H, P, N)
            h = dA * h + dBx
            ys.append(torch.einsum("bhpn,bn->bhp", h, Cm[:, t]) + self.D[None, :, None] * x[:, t])
        y = torch.stack(ys, dim=1).reshape(B, L, self.d_inner)
        y = y * F.silu(z)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-5) * self.norm_w   # gated RMSNorm
        return self.out_proj(y)


class Mamba2Regressor(nn.Module):
    """Mamba-2 regressor for turbofan RUL: embedding -> residual SSD blocks -> LayerNorm -> MLP head on the last step."""

    def __init__(self, n_sensors: int = 14, d_model: int = 48, n_layers: int = 2, d_state: int = 16, headdim: int = 16):
        super().__init__()
        self.embedding = nn.Linear(n_sensors, d_model)
        self.blocks = nn.ModuleList([SSDBlock(d_model, d_state=d_state, headdim=headdim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        for block in self.blocks:
            h = h + block(h)
        return self.head(self.norm(h)[:, -1]).squeeze(-1)


# ==============================================================================
# 5. Teacher Model: Attention-BiLSTM
# ==============================================================================
class AttentionBiLSTMTeacher(nn.Module):
    """High-capacity Attention-BiLSTM Teacher model.

    Exposes hidden trajectories and temporal attention distribution for distillation.
    """

    def __init__(
        self,
        n_sensors: int = 14,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden = hidden
        self.hidden_dim = hidden * 2
        self.bilstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden,
            num_layers=layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attn_dense = nn.Linear(hidden * 2, 32)
        self.attn_v = nn.Linear(32, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rul_pred, _, _ = self.forward_with_internals(x)
        return rul_pred

    def forward_with_internals(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns:
            rul_pred: (B,) predicted cycles
            h_traj:   (B, T, 2*hidden) hidden state trajectory
            attn_weights: (B, T) normalized attention weights
        """
        h_traj, _ = self.bilstm(x)  # (B, T, 2*hidden)
        u = torch.tanh(self.attn_dense(h_traj))  # (B, T, 32)
        scores = self.attn_v(u).squeeze(-1)      # (B, T)
        attn_weights = F.softmax(scores, dim=-1) # (B, T)

        context = torch.sum(h_traj * attn_weights.unsqueeze(-1), dim=1) # (B, 2*hidden)
        rul_pred = self.head(context).squeeze(-1)
        return rul_pred, h_traj, attn_weights


# ==============================================================================
# 6. Spiking Neural Network (SNN) Student: Parametric LIF with Surrogate Gradient
# ==============================================================================
class FastSigmoidSurrogate(torch.autograd.Function):
    """Heaviside forward; smooth surrogate derivative backward.

    mode 0 = fast sigmoid  1/(1+a|u|)^2      (Zenke & Ganguli, SuperSpike)
    mode 1 = arctan        a/2 / (1+(pi/2 a u)^2)
    mode 2 = triangle      a * max(0, 1-a|u|)
    """

    @staticmethod
    def forward(ctx, u, alpha=4.0, mode=0):
        ctx.save_for_backward(u)
        ctx.alpha, ctx.mode = alpha, mode
        return (u >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (u,) = ctx.saved_tensors
        a = ctx.alpha
        if ctx.mode == 0:
            g = 1.0 / (1.0 + a * torch.abs(u)) ** 2
        elif ctx.mode == 1:
            g = (a / 2.0) / (1.0 + (math.pi / 2.0 * a * u) ** 2)
        else:
            g = a * torch.clamp(1.0 - a * torch.abs(u), min=0.0)
        return grad_output * g, None, None


SURROGATES = {"fast_sigmoid": 0, "atan": 1, "triangle": 2}


class RecurrentLIFLayer(nn.Module):
    """(Recurrent) Leaky Integrate-and-Fire layer with learnable beta and V_th (PLIF-style).

    v[t] = beta * v[t-1] + W_in x[t] (+ W_rec s[t-1]) - V_th * s[t-1]     (soft reset)
    s[t] = H(v[t] - V_th)
    """

    def __init__(self, in_features: int, hidden_features: int, surrogate_alpha: float = 4.0,
                 surrogate: str = "fast_sigmoid", recurrent: bool = True,
                 learn_beta: bool = True, learn_vth: bool = True):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.surrogate_alpha = surrogate_alpha
        self.mode = SURROGATES[surrogate]
        self.recurrent = recurrent

        self.w_in = nn.Linear(in_features, hidden_features)
        self.w_rec = nn.Linear(hidden_features, hidden_features, bias=False) if recurrent else None

        # beta = sigmoid(theta_beta) in (0,1); init sigmoid(1.0) ~ 0.73
        self.theta_beta = nn.Parameter(torch.full((hidden_features,), 1.0), requires_grad=learn_beta)
        # V_th = softplus(theta_v) > 0; init ~ 1.0
        self.theta_v = nn.Parameter(torch.full((hidden_features,), 0.5413), requires_grad=learn_vth)

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.theta_beta)

    @property
    def v_th(self) -> torch.Tensor:
        return F.softplus(self.theta_v)

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_seq (B,T,in) -> spikes (B,T,H), voltages (B,T,H)."""
        B, T, _ = x_seq.shape
        v = torch.zeros(B, self.hidden_features, device=x_seq.device)
        s = torch.zeros(B, self.hidden_features, device=x_seq.device)
        beta, v_th = self.beta, self.v_th
        i_all = self.w_in(x_seq)  # input currents for every step at once
        spikes_list, voltages_list = [], []
        for t in range(T):
            i = i_all[:, t]
            if self.recurrent:
                i = i + self.w_rec(s)
            v = beta * v + i - s * v_th
            s = FastSigmoidSurrogate.apply(v - v_th, self.surrogate_alpha, self.mode)
            voltages_list.append(v)
            spikes_list.append(s)
        return torch.stack(spikes_list, dim=1), torch.stack(voltages_list, dim=1)


class SpikeDistillStudent(nn.Module):
    """SpikeDistill-RUL: 2-layer (recurrent) LIF spiking student with a membrane-potential readout."""

    def __init__(self, n_sensors: int = 14, hidden: int = 48, surrogate_alpha: float = 4.0,
                 surrogate: str = "fast_sigmoid", recurrent: bool = True,
                 learn_beta: bool = True, learn_vth: bool = True, readout: str = "attn"):
        super().__init__()
        self.hidden = hidden
        self.n_sensors = n_sensors
        self.readout = readout  # "attn" = learned attention pooling; "mean" = plain average of v2 over time (cheaper)
        kw = dict(surrogate_alpha=surrogate_alpha, surrogate=surrogate, recurrent=recurrent,
                  learn_beta=learn_beta, learn_vth=learn_vth)
        self.lif1 = RecurrentLIFLayer(n_sensors, hidden, **kw)
        self.lif2 = RecurrentLIFLayer(hidden, hidden, **kw)
        self.attn_dense = nn.Linear(hidden, 16)
        self.attn_v = nn.Linear(16, 1, bias=False)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self.rate1 = self.rate2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_internals(x)[0]

    def forward_with_internals(self, x: torch.Tensor):
        """Returns rul (B,), v2 (B,T,H), s2 (B,T,H), attn (B,T), avg_spike_rate (scalar)."""
        s1, _ = self.lif1(x)
        s2, v2 = self.lif2(s1)
        if self.readout == "mean":
            attn = torch.full(v2.shape[:2], 1.0 / v2.size(1), device=v2.device)
        else:
            u = torch.tanh(self.attn_dense(v2))
            attn = F.softmax(self.attn_v(u).squeeze(-1), dim=-1)
        context = torch.sum(v2 * attn.unsqueeze(-1), dim=1)
        rul = self.head(context).squeeze(-1)
        self.rate1, self.rate2 = s1.mean(), s2.mean()
        avg = (s1.sum() + s2.sum()) / (s1.numel() + s2.numel())
        return rul, v2, s2, attn, avg

    def ops(self, T: int, rate1: float, rate2: float) -> Dict[str, float]:
        """Per-window operation counts: dense MACs vs event-driven accumulates (AC).

        Layer 1 sees analog sensor values -> dense MACs on W_in1. Everything fed by spikes is an AC
        and only happens when the presynaptic neuron fired. The membrane readout is analog -> MACs.
        """
        H, n = self.hidden, self.n_sensors
        r1 = H * H if self.lif1.recurrent else 0
        r2 = H * H if self.lif2.recurrent else 0
        macs = T * n * H                            # W_in1 on analog input
        macs += 2 * T * H                           # beta * v decay, both layers
        if self.readout != "mean":
            macs += T * (H * 16 + 16) + T * H       # attention scorer + weighted sum
        macs += H * 32 + 32                         # head
        ac = T * (r1 * rate1 + H * H * rate1 + r2 * rate2)  # W_rec1, W_in2 (fed by s1), W_rec2
        return {"macs": macs, "acs": ac}


# ==============================================================================
# 7. Vanilla-KD Small LSTM (Baseline Distillation)
# ==============================================================================
class VanillaKDSmallLSTM(nn.Module):
    """Tiny 1-layer LSTM student for baseline vanilla knowledge distillation."""

    def __init__(self, n_sensors: int = 14, hidden: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


# ==============================================================================
# 8. Extra sequence baselines: TCN, Transformer, Bi-Mamba, generic attention-pool teachers
# ==============================================================================
class TCNRegressor(nn.Module):
    """Causal dilated temporal convolutions with residuals (Bai et al., 2018)."""

    def __init__(self, n_sensors: int = 14, channels: int = 32, kernel: int = 3, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.kernel = kernel
        self.dilations = dilations
        self.inp = nn.Conv1d(n_sensors, channels, 1)
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, kernel, dilation=d) for d in dilations])
        self.head = nn.Sequential(nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for conv, d in zip(self.convs, self.dilations):
            pad = (self.kernel - 1) * d
            h = h + F.relu(conv(F.pad(h, (pad, 0))))  # left-pad only = causal
        return self.head(h[:, :, -1]).squeeze(-1)


class TransformerEncoderSeq(nn.Module):
    def __init__(self, n_sensors, d_model=32, heads=4, layers=2, ff=64, window=30, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(n_sensors, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_dim = d_model

    def forward(self, x):
        return self.enc(self.embed(x) + self.pos[:, : x.size(1)])


class TransformerRegressor(nn.Module):
    def __init__(self, n_sensors: int = 14, **kw):
        super().__init__()
        self.encoder = TransformerEncoderSeq(n_sensors, **kw)
        d = self.encoder.out_dim
        self.head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.encoder(x).mean(dim=1)).squeeze(-1)


class BiMambaEncoder(nn.Module):
    """Forward Mamba stack + Mamba stack on the time-reversed window; features concatenated."""

    def __init__(self, n_sensors, d_model=32, n_layers=2, d_state=16, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.fwd = MambaRegressor.__new__(MambaRegressor)
        nn.Module.__init__(self.fwd)
        self.fwd.embedding = nn.Linear(n_sensors, d_model)
        self.fwd.blocks = nn.ModuleList([S6Block(d_model, d_state) for _ in range(n_layers)])
        self.fwd.norm = nn.LayerNorm(d_model)
        if bidirectional:
            self.bwd = MambaRegressor.__new__(MambaRegressor)
            nn.Module.__init__(self.bwd)
            self.bwd.embedding = nn.Linear(n_sensors, d_model)
            self.bwd.blocks = nn.ModuleList([S6Block(d_model, d_state) for _ in range(n_layers)])
            self.bwd.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model * (2 if bidirectional else 1)

    @staticmethod
    def _run(m, x):
        h = m.embedding(x)
        for b in m.blocks:
            h = h + b(h)
        return m.norm(h)

    def forward(self, x):
        f = self._run(self.fwd, x)
        if not self.bidirectional:
            return f
        b = self._run(self.bwd, x.flip(1)).flip(1)  # re-align to original time order
        return torch.cat([f, b], dim=-1)


class BiMambaRegressor(nn.Module):
    def __init__(self, n_sensors: int = 14, d_model: int = 32, n_layers: int = 2, d_state: int = 16):
        super().__init__()
        self.encoder = BiMambaEncoder(n_sensors, d_model, n_layers, d_state, True)
        d = self.encoder.out_dim
        self.head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.encoder(x)
        d = h.size(-1) // 2
        # forward half at last step + backward half at first step (it has now seen the whole window)
        return self.head(torch.cat([h[:, -1, :d], h[:, 0, d:]], dim=-1)).squeeze(-1)


class AttnPoolTeacher(nn.Module):
    """Teacher = sequence encoder (transformer | bimamba) + additive temporal attention pooling.

    Same distillation interface as AttentionBiLSTMTeacher.
    """

    def __init__(self, n_sensors: int = 14, backbone: str = "transformer", dropout: float = 0.2, window: int = 30):
        super().__init__()
        if backbone == "transformer":
            self.encoder = TransformerEncoderSeq(n_sensors, d_model=64, heads=4, layers=3, ff=128, window=window)
        elif backbone == "bimamba":
            self.encoder = BiMambaEncoder(n_sensors, d_model=32, n_layers=2, d_state=16)
        else:
            raise ValueError(backbone)
        d = self.hidden_dim = self.encoder.out_dim
        self.attn_dense = nn.Linear(d, 32)
        self.attn_v = nn.Linear(32, 1, bias=False)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        return self.forward_with_internals(x)[0]

    def forward_with_internals(self, x):
        h = self.encoder(x)
        attn = F.softmax(self.attn_v(torch.tanh(self.attn_dense(h))).squeeze(-1), dim=-1)
        ctx = torch.sum(h * attn.unsqueeze(-1), dim=1)
        return self.head(ctx).squeeze(-1), h, attn


class SmallGRU(nn.Module):
    """Tiny non-spiking GRU student (vanilla-KD control)."""

    def __init__(self, n_sensors: int = 14, hidden: int = 16):
        super().__init__()
        self.gru = nn.GRU(n_sensors, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)



# ==============================================================================
# 9. Hybrid architectures added to try to beat the plain baselines
# ==============================================================================
class AttnPool(nn.Module):
    """Additive attention pooling over time: score each step, softmax, weighted mean (Bahdanau-style)."""

    def __init__(self, d: int, a: int = 32):
        super().__init__()
        self.dense = nn.Linear(d, a)
        self.v = nn.Linear(a, 1, bias=False)

    def forward(self, h: torch.Tensor):
        w = F.softmax(self.v(torch.tanh(self.dense(h))).squeeze(-1), dim=-1)  # (B, T)
        return torch.sum(h * w.unsqueeze(-1), dim=1), w


class TCNAttnRegressor(nn.Module):
    """Causal dilated TCN with residual blocks, read out by attention over ALL steps (not just the last)."""

    def __init__(self, n_sensors: int = 14, channels: int = 32, kernel: int = 3, dilations=(1, 2, 4, 8), dropout: float = 0.1):
        super().__init__()
        self.kernel, self.dilations = kernel, tuple(dilations)
        self.inp = nn.Conv1d(n_sensors, channels, 1)
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, kernel, dilation=d) for d in self.dilations])
        self.drop = nn.Dropout(dropout)
        self.pool = AttnPool(channels)
        self.head = nn.Sequential(nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for conv, d in zip(self.convs, self.dilations):
            pad = (self.kernel - 1) * d
            h = h + self.drop(F.relu(conv(F.pad(h, (pad, 0)))))
        ctx, _ = self.pool(h.transpose(1, 2))
        return self.head(ctx).squeeze(-1)


class CNNBiGRUAttn(nn.Module):
    """Conv front end -> 2-layer BiGRU -> attention pooling -> MLP head."""

    def __init__(self, n_sensors: int = 14, channels: int = 32, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(n_sensors, channels, 3, padding=1)
        self.gru = nn.GRU(channels, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttnPool(2 * hidden)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        h, _ = self.gru(h)
        ctx, _ = self.pool(h)
        return self.head(ctx).squeeze(-1)


class ConvTransformer(nn.Module):
    """Conv stem (local patterns) -> Transformer encoder (global) -> attention pooling."""

    def __init__(self, n_sensors: int = 14, d_model: int = 32, heads: int = 4, layers: int = 2, ff: int = 64,
                 max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Conv1d(n_sensors, d_model, 3, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.pool = AttnPool(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = F.gelu(self.stem(x.transpose(1, 2))).transpose(1, 2)
        h = self.enc(h + self.pos[:, : h.size(1)])
        ctx, _ = self.pool(h)
        return self.head(ctx).squeeze(-1)

# ==============================================================================
# Model Factory & Utilities
# ==============================================================================
def create_model(name: str, n_sensors: int = 14, **kwargs) -> nn.Module:
    """Convenience factory to instantiate any model by name."""
    name_lower = name.lower().replace("-", "").replace("_", "")
    if name_lower == "widenn":
        return WideNN(n_sensors=n_sensors, **kwargs)
    elif name_lower == "deepnn":
        return DeepNN(n_sensors=n_sensors, **kwargs)
    elif name_lower in ("rnn", "gru", "lstm", "bilstm"):
        return RecurrentRegressor(n_sensors=n_sensors, kind=name_lower, **kwargs)
    elif name_lower == "cnnlstm":
        return CNNLSTMRegressor(n_sensors=n_sensors, **kwargs)
    elif name_lower == "mamba":
        return MambaRegressor(n_sensors=n_sensors, **kwargs)
    elif name_lower == "mamba2":
        return Mamba2Regressor(n_sensors=n_sensors, **kwargs)
    elif name_lower == "bimamba":
        return BiMambaRegressor(n_sensors=n_sensors, **kwargs)
    elif name_lower == "tcn":
        return TCNRegressor(n_sensors=n_sensors, **kwargs)
    elif name_lower == "transformer":
        return TransformerRegressor(n_sensors=n_sensors, **kwargs)
    elif name_lower in ("transformerteacher", "bimambateacher"):
        return AttnPoolTeacher(n_sensors=n_sensors, backbone=name_lower.replace("teacher", ""), **kwargs)
    elif name_lower == "tcnattn":
        return TCNAttnRegressor(n_sensors=n_sensors, **kwargs)
    elif name_lower == "cnnbigruattn":
        return CNNBiGRUAttn(n_sensors=n_sensors, **kwargs)
    elif name_lower == "convtransformer":
        return ConvTransformer(n_sensors=n_sensors, **kwargs)
    elif name_lower == "smallgru":
        return SmallGRU(n_sensors=n_sensors, **kwargs)
    elif name_lower in ("attentionbilstm", "teacher"):
        return AttentionBiLSTMTeacher(n_sensors=n_sensors, **kwargs)
    elif name_lower in ("spikedistill", "student", "snn"):
        return SpikeDistillStudent(n_sensors=n_sensors, **kwargs)
    elif name_lower in ("smalllstm", "vanillakd"):
        return VanillaKDSmallLSTM(n_sensors=n_sensors, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")


def count_params(model: nn.Module) -> int:
    """Returns number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
