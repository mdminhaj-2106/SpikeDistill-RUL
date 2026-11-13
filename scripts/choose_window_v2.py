"""Choose the window length (time steps) on VALIDATION data only, with a stated rule. Output: results/window_choice_v2.json

Data      results/suite/base__*__sw<W>__*.json : GRU, LSTM, TCN and the attention-BiLSTM, 4 subsets, 3 seeds, W in SWEEP_W.
Metric    RMSE on the TEST-LIKE validation set (every cut of every held-out engine with true RUL in [1,125], padded like test).
          That set depends only on the engine split, so each window is scored on exactly the same cuts: windows are comparable.
Cell      one (model, subset, seed) triple = 48 cells. The same cell is trained at every window length.
Rule      one-standard-error rule: let W* be the window with the lowest mean validation RMSE; choose the SHORTEST window whose
          paired difference to W* (mean over cells) is below one standard error of that difference. Shorter windows cost less
          compute, energy and padding, and are what a streaming device would have to buffer.
Test RMSE is printed beside it as a sanity check only; it is never used to choose.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from aggregate import load  # noqa: E402

df = load(); df["tag"] = df["tag"].fillna("")
b = df[(df.kind == "base") & df.tag.str.match(r"^sw\d+$")].copy()
b["W"] = b.tag.str[2:].astype(int)
cell = ["model", "subset", "seed"]
Ws = sorted(b.W.unique())
full = [W for W in Ws if len(b[b.W == W]) == max(len(b[b.W == w]) for w in Ws)]          # only windows with every cell finished
P = b[b.W.isin(full)].pivot_table(index=cell, columns="W", values="val_tl_RMSE")
T = b[b.W.isin(full)].pivot_table(index=cell, columns="W", values="test_RMSE_c125")
P = P.dropna()
mean = P.mean(); best = int(mean.idxmin())
tab = {}
for W in full:
    d = (P[W] - P[best]).to_numpy(); se = d.std(ddof=1) / np.sqrt(len(d)) if W != best else 0.0
    tab[int(W)] = {"val_tl_rmse": float(mean[W]), "diff_to_best": float(d.mean()), "se_diff": float(se), "test_rmse_capped": float(T[W].mean()), "n_cells": int(len(d))}
choice = int(min(W for W in full if tab[int(W)]["diff_to_best"] <= tab[int(W)]["se_diff"] + 1e-12))
per_sub = {}
for s in sorted(b.subset.unique()):
    Ps = P.xs(s, level="subset"); m = Ps.mean(); bw = int(m.idxmin())
    per_sub[s] = {"best_window": bw, "val_tl_by_window": {int(W): float(m[W]) for W in full}}
out = {"metric": "val_tl RMSE (test-like validation set)", "rule": "shortest window within one SE (paired) of the best", "best_window": best, "window": choice,
       "table": tab, "per_subset": per_sub, "windows_complete": [int(w) for w in full]}
(ROOT / "results" / "window_choice_v2.json").write_text(json.dumps(out, indent=1))
print(f"{'W':>4} {'val_tl':>8} {'diff':>7} {'se':>6} {'test(capped)':>13}")
for W, r in tab.items():
    print(f"{W:>4} {r['val_tl_rmse']:8.2f} {r['diff_to_best']:+7.2f} {r['se_diff']:6.2f} {r['test_rmse_capped']:13.2f}" + ("  <- chosen" if W == choice else ""))
print("best by mean:", best, "| chosen:", choice); print({s: v["best_window"] for s, v in per_sub.items()})
