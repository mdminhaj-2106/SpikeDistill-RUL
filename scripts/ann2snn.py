"""ANN-to-SNN conversion baseline (assignment baseline) and its accuracy-vs-simulation-time-steps curve.

1. Train a ReLU MLP on flattened windows (same data pipeline, split and seeds as every other model).
2. Convert to integrate-and-fire (IF) neurons by data-based normalisation (Diehl 2015; Rueckauer 2017):
     lambda_l = 99.9th percentile of the layer-l ReLU activations on TRAINING windows;
     W'_l = W_l * lambda_{l-1} / lambda_l,   b'_l = b_l / lambda_l,   threshold = 1, reset by subtraction.
3. Simulate T_s time steps with the (constant) analog input injected as current; the output layer integrates without spiking:
     prediction = (1/T_s) * sum_t ( W_out * lambda_L * s_L(t) + b_out ).
4. Count operations: input layer = dense MACs once, everything fed by spikes = accumulates (AC) only when a spike arrives.
Output: results/ann2snn_v2.json (protocol v2: window from window_choice_v2.json, early stopping on the test-like validation set, seeds 0-4)   (per subset, seed, T_s: RMSE official + capped, spike rate, MACs, ACs, energy pJ)
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
from rul.data import prepare
from rul.metrics import all_metrics
from rul.train import fit, predict

WIN = int(json.load(open(ROOT / "results/window_choice_v2.json"))["window"])      # protocol v2 window
TS = [4, 8, 16, 32, 64, 128, 256, 512]
E_MAC, E_AC = 4.6, 0.9


class ReLUMLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64)):
        super().__init__()
        dims = [in_dim, *hidden]
        self.hidden = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, x):
        h = x.reshape(x.size(0), -1)
        for l in self.hidden:
            h = torch.relu(l(h))
        return self.out(h).squeeze(-1)


@torch.no_grad()
def convert_and_run(net, X_norm, X, T_s):
    """Return predictions (scaled units) and operation counts for window batch X (N,T,S) over T_s steps."""
    net.eval()
    # --- normalisation factors from training activations
    h = X_norm.reshape(len(X_norm), -1); lam = [1.0]
    for l in net.hidden:
        h = torch.relu(l(h)); lam.append(float(torch.quantile(h.flatten()[:2_000_000], 0.999).clamp_min(1e-6)))
    Ws = [(l.weight * lam[i] / lam[i + 1], l.bias / lam[i + 1]) for i, l in enumerate(net.hidden)]
    Wo, bo = net.out.weight * lam[-1], net.out.bias
    x = X.reshape(len(X), -1)
    v = [torch.zeros(len(X), W.shape[0]) for W, _ in Ws]
    out = torch.zeros(len(X)); spikes = [0.0] * len(Ws); acs = 0.0
    I1 = x @ Ws[0][0].T + Ws[0][1]                        # input current is constant: one dense pass
    macs = float(x.shape[1] * Ws[0][0].shape[0])
    for t in range(T_s):
        inp = None
        for i, (W, b) in enumerate(Ws):
            cur = I1 if i == 0 else inp @ W.T + b
            if i > 0:
                acs += float(inp.sum(dim=1).mean()) * W.shape[0]        # one add per out-neuron per incoming spike
            v[i] = v[i] + cur
            s = (v[i] >= 1.0).float(); v[i] = v[i] - s
            spikes[i] += float(s.mean()); inp = s
        acs += float(inp.sum(dim=1).mean())                            # output layer: one add per incoming spike
        out = out + (inp @ Wo.T).squeeze(-1) + bo
    pred = out / T_s
    rates = [sp / T_s for sp in spikes]
    return pred.numpy(), {"macs": macs, "acs": acs, "rates": rates}


def main():
    torch.set_num_threads(2)
    res = []
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        for seed in range(5):
            torch.manual_seed(seed); np.random.seed(seed)
            d = prepare(sub, seed=seed, window=WIN, condition_aware=sub in ("FD002", "FD004"))
            S = len(d["sensors"]); net = ReLUMLP(WIN * S)
            r = fit(net, d, epochs=40, patience=8, device="cpu", verbose=False, y_scale=125.0, select="tl")
            ann_pred = predict(net, d["X_test"], "cpu", y_scale=125.0).clip(0, 125)
            y = d["y_test"]; yc = np.minimum(y, 125)
            rec = {"subset": sub, "seed": seed, "params": sum(p.numel() for p in net.parameters()),
                   "ann": {"rmse": float(np.sqrt(np.mean((ann_pred - y) ** 2))), "rmse_c125": float(np.sqrt(np.mean((ann_pred - yc) ** 2))),
                           "macs": float(sum(l.weight.numel() for l in list(net.hidden) + [net.out])), "val_rmse": r["val"]["RMSE"]},
                   "snn": {}}
            Xn = torch.as_tensor(d["X_train"][:5000]); Xt = torch.as_tensor(d["X_test"])
            for T_s in TS:
                p, ops = convert_and_run(net, Xn, Xt, T_s)
                p = (p * 125.0).clip(0, 125)
                rec["snn"][str(T_s)] = {"rmse": float(np.sqrt(np.mean((p - y) ** 2))), "rmse_c125": float(np.sqrt(np.mean((p - yc) ** 2))),
                                         "macs": ops["macs"], "acs": ops["acs"], "rates": ops["rates"], "energy_pj": ops["macs"] * E_MAC + ops["acs"] * E_AC}
            res.append(rec)
            print(sub, seed, "ANN", round(rec["ann"]["rmse"], 2), "SNN", {k: round(v["rmse"], 2) for k, v in rec["snn"].items()}, flush=True)
    (ROOT / "results" / "ann2snn_v2.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
