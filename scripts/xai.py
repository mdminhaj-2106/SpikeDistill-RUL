"""Explainability for the spiking student and the attention-BiLSTM teacher on FD001 (seed 0).

Five methods (the assignment asks for at least four):
  1. Integrated Gradients per (cycle, sensor) with a completeness check
  2. Exact Shapley values per sensor (14 players, 2^14 coalitions) and per 5-cycle time block (6 players)
  3. Attention maps (teacher vs student), grouped by true RUL
  4. Spike-raster attribution for layer 2 of the student
  5. Permutation importance of sensors (increase in test RMSE when a sensor is shuffled across test windows)
Plus the rank agreement between the sensor rankings of methods 1, 2 and 5.

Run:  python scripts/xai.py            (trains + saves the student checkpoint if missing, then writes results/xai/xai.json)
All attributions explain the MODEL, not the engine physics; they are associations, not causal proof.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rul.data import prepare  # noqa: E402
from rul.distill import fit_distillation  # noqa: E402
from rul.models import create_model  # noqa: E402
from rul.metrics import all_metrics  # noqa: E402

OUT = ROOT / "results" / "xai"
SUBSET, SEED, SCALE = "FD001", 0, 125.0
WIN = int(json.load(open(ROOT / "results/window_choice_v2.json"))["window"])      # protocol v2 window (60)
DESC = {"s1": "T2 fan inlet temp", "s2": "T24 LPC outlet temp", "s3": "T30 HPC outlet temp", "s4": "T50 LPT outlet temp",
        "s5": "P2 fan inlet pressure", "s6": "P15 bypass pressure", "s7": "P30 HPC outlet pressure", "s8": "Nf fan speed",
        "s9": "Nc core speed", "s10": "epr", "s11": "Ps30 HPC static pressure", "s12": "phi fuel/Ps30", "s13": "NRf corrected fan speed",
        "s14": "NRc corrected core speed", "s15": "BPR bypass ratio", "s16": "farB fuel-air ratio", "s17": "htBleed enthalpy",
        "s18": "Nf_dmd", "s19": "PCNfR_dmd", "s20": "W31 HPT coolant bleed", "s21": "W32 LPT coolant bleed"}


def f_cycles(model, x):
    """Model output in cycles (the network is trained on RUL/125)."""
    return model(x) * SCALE


def integrated_gradients(model, X, base, steps=32):
    """IG_i = (x_i - b_i) * mean_k dF/dx_i(b + k/steps (x - b)). X: (N,T,S) tensor, base: (T,S). Returns (N,T,S) and completeness gap."""
    model.eval()
    X = X.clone(); b = base.expand_as(X)
    grads = torch.zeros_like(X)
    for k in range(1, steps + 1):
        z = (b + (k / steps) * (X - b)).requires_grad_(True)
        out = f_cycles(model, z).sum()
        grads += torch.autograd.grad(out, z)[0]
    ig = (X - b) * grads / steps
    with torch.no_grad():
        gap = (ig.sum(dim=(1, 2)) - (f_cycles(model, X) - f_cycles(model, b)))
    return ig.detach(), gap.detach()


def shapley_exact(value_fn, n):
    """Exact Shapley values from a value function over all 2^n coalitions. value_fn(masks: (M,n) bool) -> (M,) ; returns (n,)."""
    masks = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)
    v = value_fn(masks)
    size = masks.sum(1)
    from math import factorial
    w = np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])
    phi = np.zeros(n)
    idx = np.arange(2 ** n)
    for i in range(n):
        without = idx[~masks[:, i]]
        phi[i] = np.sum(w[size[without]] * (v[without | (1 << i)] - v[without]))
    return phi


@torch.no_grad()
def shap_sensor_and_block(model, X, base, blocks=6, chunk=8192):
    """Mean |Shapley| per sensor (players=sensors) and per time block (players=blocks), averaged over the windows in X."""
    model.eval()
    N, T, S = X.shape
    bl = T // blocks
    sens = np.zeros(S); blk = np.zeros(blocks); signed_s = np.zeros(S)
    for n in range(N):
        x = X[n]
        def val_sensor(masks):
            out = []
            for i in range(0, len(masks), chunk):
                m = torch.as_tensor(masks[i:i + chunk], dtype=torch.float32)            # (M,S) 1 = sensor present
                z = base.unsqueeze(0) * (1 - m[:, None, :]) + x.unsqueeze(0) * m[:, None, :]
                out.append(f_cycles(model, z))
            return torch.cat(out).numpy()
        def val_block(masks):
            m = torch.as_tensor(masks, dtype=torch.float32).repeat_interleave(bl, dim=1)  # (M,T)
            z = base.unsqueeze(0) * (1 - m[:, :, None]) + x.unsqueeze(0) * m[:, :, None]
            return f_cycles(model, z).numpy()
        ps = shapley_exact(val_sensor, S); pb = shapley_exact(val_block, blocks)
        sens += np.abs(ps); signed_s += ps; blk += np.abs(pb)
    return sens / N, blk / N, signed_s / N


@torch.no_grad()
def permutation_importance(model, X, y, repeats=10, seed=0):
    rng = np.random.default_rng(seed)
    model.eval()
    def rmse(Xn):
        p = (f_cycles(model, torch.as_tensor(Xn)).numpy()).clip(0, 125)
        return float(np.sqrt(np.mean((p - y) ** 2)))
    base = rmse(X)
    imp = np.zeros(X.shape[2])
    for j in range(X.shape[2]):
        d = []
        for _ in range(repeats):
            Xp = X.copy(); Xp[:, :, j] = Xp[rng.permutation(len(X)), :, j]
            d.append(rmse(Xp) - base)
        imp[j] = np.mean(d)
    return imp, base


def rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(SEED); np.random.seed(SEED)
    d = prepare(SUBSET, seed=SEED, window=WIN)
    S = len(d["sensors"]); names = d["sensors"]
    teacher = create_model("teacher", S)
    teacher.load_state_dict(torch.load(ROOT / "results/suite/ckpt" / f"teacher_{SUBSET}_s{SEED}_v2.pt", map_location="cpu"))
    teacher.eval()
    ck = OUT / f"student_{SUBSET}_s{SEED}_v2.pt"
    student = create_model("student", S)
    if ck.exists():
        student.load_state_dict(torch.load(ck, map_location="cpu"))
    else:
        t = time.time()
        r = fit_distillation(student, teacher, d, epochs=40, patience=8, device="cpu", verbose=False, select="tl")
        torch.save(student.state_dict(), ck)
        (OUT / "student_train.json").write_text(json.dumps({"test": r["test"], "epochs_run": r["epochs_run"], "seconds": time.time() - t,
                                                              "rate1": r["rate1"], "rate2": r["rate2"]}))
    student.eval()

    X = torch.as_tensor(d["X_test"]); y = d["y_test"]
    base = torch.as_tensor(d["X_train"].mean(axis=(0, 1)), dtype=torch.float32).expand(X.shape[1], S).contiguous()  # mean healthy-ish engine
    res = {"sensors": names, "desc": [DESC.get(n, n) for n in names], "y_test": y.tolist()}
    with torch.no_grad():
        for nm, m in (("teacher", teacher), ("student", student)):
            p = f_cycles(m, X).numpy().clip(0, 125)
            res[f"{nm}_test"] = all_metrics(y, p)
            res[f"{nm}_pred"] = p.tolist()

    summary = {}
    for nm, m in (("teacher", teacher), ("student", student)):
        t0 = time.time()
        ig, gap = integrated_gradients(m, X, base)
        sens_s, blk_s, sgn_s = shap_sensor_and_block(m, X, base)
        perm, base_rmse = permutation_importance(m, X.numpy(), y)
        ig_sensor = ig.abs().sum(dim=1).mean(0).numpy()      # sum over cycles of |IG|, mean over windows
        ig_time = ig.abs().sum(dim=2).mean(0).numpy()        # per cycle
        ig_signed_sensor = ig.sum(dim=1).mean(0).numpy()
        summary[nm] = {"ig_sensor": ig_sensor.tolist(), "ig_signed_sensor": ig_signed_sensor.tolist(), "ig_time": ig_time.tolist(),
                       "ig_completeness_gap_mean_abs": float(gap.abs().mean()), "ig_completeness_gap_rel": float(gap.abs().mean() / (f_cycles(m, X).detach() - f_cycles(m, base.unsqueeze(0)).detach()).abs().mean()),
                       "shap_sensor": sens_s.tolist(), "shap_signed_sensor": sgn_s.tolist(), "shap_block": blk_s.tolist(),
                       "perm": perm.tolist(), "perm_base_rmse": base_rmse,
                       "rank_corr_ig_shap": rank_corr(ig_sensor, sens_s), "rank_corr_ig_perm": rank_corr(ig_sensor, perm),
                       "rank_corr_shap_perm": rank_corr(sens_s, perm), "seconds": time.time() - t0}
        print(nm, "done", round(time.time() - t0), "s", {k: round(v, 3) for k, v in summary[nm].items() if k.startswith("rank") or k.startswith("ig_comp")}, flush=True)
    res["methods"] = summary

    # attention maps by true-RUL group
    with torch.no_grad():
        _, _, a_t = teacher.forward_with_internals(X)
        _, v2, s2, a_s, _ = student.forward_with_internals(X)
    grp = {"RUL < 30": y < 30, "30 to 80": (y >= 30) & (y < 80), "RUL >= 80": y >= 80}
    res["attention"] = {g: {"teacher": a_t[torch.as_tensor(msk)].mean(0).tolist(), "student": a_s[torch.as_tensor(msk)].mean(0).tolist(), "n": int(msk.sum())}
                        for g, msk in grp.items() if msk.sum() > 0}

    # spike-raster attribution for one low-RUL and one high-RUL test engine
    def raster(i):
        xi = X[i:i + 1]
        _, v2i, s2i, ai, _ = student.forward_with_internals(xi)
        ctx = (v2i * ai.unsqueeze(-1)).sum(1).detach().requires_grad_(True)
        out = student.head(ctx).squeeze(-1)
        g = torch.autograd.grad(out, ctx)[0]                                   # dRUL/dcontext (1,H)
        attr = (g.unsqueeze(1) * ai.unsqueeze(-1) * v2i).detach()[0]           # (T,H) first-order contribution of each (cycle, neuron)
        return {"engine_idx": int(i), "true_rul": float(y[i]), "pred": float(125 * out.item()), "spikes": s2i.detach()[0].numpy().tolist(),
                "attr": attr.numpy().tolist(), "first_order_sum": float(125 * attr.sum()), "rate": float(s2i.mean())}
    lo, hi = int(np.argmin(y)), int(np.argmax(y))
    res["raster"] = {"low": raster(lo), "high": raster(hi)}
    (OUT / "xai.json").write_text(json.dumps(res))
    print("wrote", OUT / "xai.json")


if __name__ == "__main__":
    main()
