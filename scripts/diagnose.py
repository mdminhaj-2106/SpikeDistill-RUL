"""Root-cause audit: why do very different architectures (and the teacher) land within noise of simple baselines?

Each block tests one candidate cause against the stored results / data and writes results/diagnostics.json.
  A coverage        which (model, subset, window) cells exist and with how many seeds
  B undertraining   did runs hit the epoch cap, or stop with validation still improving?
  C resolution      how small a RMSE difference can 100-250 test engines resolve? (bootstrap over engines)
  D shared error    do different models make the SAME mistakes on the same engines? (residual correlation, ensemble gain)
  E padding         do test engines shorter than the window (padded) behave differently from the rest?
  F naive baselines constant, ridge and gradient boosting on hand-made window features
Run: .venv/bin/python scripts/diagnose.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from aggregate import load  # noqa: E402
from rul.data import load_raw, prepare  # noqa: E402

SUBS = ["FD001", "FD002", "FD003", "FD004"]
OUT = {}
TAG = sys.argv[1] if len(sys.argv) > 1 else "w50"          # run tag to audit: "w50" (protocol v1) or "v2" (protocol v2)
WIN = int(sys.argv[2]) if len(sys.argv) > 2 else 50        # its window length
NSEED = 8 if TAG == "v2" else 3
SEEDS_ENS = (0, 1, 2, 3, 4) if TAG == "v2" else (0, 1, 2)   # same seeds for every model
SKIP = ("transformerteacher",)
files = sorted((ROOT / "results/suite").glob("base__*.json"))
recs = [json.load(open(f)) for f in files]
rows = [r for r in recs if "test_pred" in r]
rmse = lambda p, y: float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))

# ---------------------------------------------------------------- A coverage
cov = {}
for r in recs:
    key = f"{r['model']}|{r.get('tag', '')}"
    cov.setdefault(key, {}).setdefault(r["subset"], set()).add(r["seed"])
OUT["coverage"] = {k: {s: len(v.get(s, ())) for s in SUBS} for k, v in sorted(cov.items())}

# ---------------------------------------------------------------- B undertraining
ut = {}
for r in recs:
    if r.get("tag", "") not in (("", "w50") if TAG == "w50" else (TAG,)):
        continue
    h = r["val_rmse_history"]
    best = int(np.argmin(h)) + 1
    ut.setdefault(r["model"], []).append({"hit_cap": r["epochs_run"] >= r["epochs"], "best_is_last_quarter": best > 0.75 * r["epochs"],
                                          "gap": r["train_rmse_history"][best - 1] - h[best - 1], "best_epoch": best})
OUT["undertraining"] = {m: {"n": len(v), "hit_epoch_cap": float(np.mean([x["hit_cap"] for x in v])), "best_epoch_median": float(np.median([x["best_epoch"] for x in v])),
                            "train_minus_val_rmse": float(np.mean([x["gap"] for x in v]))} for m, v in ut.items()}

# ---------------------------------------------------------------- C resolution (bootstrap over test engines)
rng = np.random.default_rng(0)
res = {}
for s in SUBS:
    cand = [r for r in rows if r["subset"] == s and r.get("tag") == TAG and r["seed"] == 0]
    if len(cand) < 2:
        continue
    y = np.asarray(cand[0]["y_test"], float); n = len(y)
    P = {r["model"]: np.asarray(r["test_pred"], float) for r in cand}
    e2 = {m: (p - y) ** 2 for m, p in P.items()}
    se = {}
    for m, v in e2.items():
        b = [np.sqrt(v[rng.integers(0, n, n)].mean()) for _ in range(1000)]
        se[m] = float(np.std(b))
    ms = list(P)
    diffs = []
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            d = []
            for _ in range(400):
                idx = rng.integers(0, n, n)
                d.append(np.sqrt(e2[ms[i]][idx].mean()) - np.sqrt(e2[ms[j]][idx].mean()))
            diffs.append(float(np.std(d)))
    res[s] = {"n_engines": n, "rmse_se_median": float(np.median(list(se.values()))), "paired_diff_se_median": float(np.median(diffs)),
              "min_detectable_diff_95": float(1.96 * np.median(diffs))}
OUT["resolution"] = res

# ---------------------------------------------------------------- D shared error
shared = {}
for s in SUBS:
    for seed in SEEDS_ENS:
        cand = [r for r in rows if r["subset"] == s and r.get("tag") == TAG and r["seed"] == seed and r["model"] not in SKIP]
        if len(cand) < 5:
            continue
        y = np.asarray(cand[0]["y_test"], float)
        E = np.stack([np.asarray(r["test_pred"], float) - np.minimum(y, 125) for r in cand])
        C = np.corrcoef(E); iu = np.triu_indices(len(cand), 1)
        ens = rmse(np.mean([np.asarray(r["test_pred"], float) for r in cand], 0), np.minimum(y, 125))
        indiv = [rmse(r["test_pred"], np.minimum(y, 125)) for r in cand]
        shared.setdefault(s, []).append({"mean_residual_corr": float(C[iu].mean()), "ensemble_rmse": ens, "best_single": float(min(indiv)), "mean_single": float(np.mean(indiv)), "n_models": len(cand)})
OUT["shared_error"] = {s: {k: float(np.mean([x[k] for x in v])) for k in v[0]} for s, v in shared.items()}

# ---------------------------------------------------------------- E padding (test engines shorter than the window)
pad = {}
for s in SUBS:
    _, test, true_rul = load_raw(s)
    lens = test.groupby("engine").size().sort_index().to_numpy()
    for tag, W in ((("", 30), ("w50", 50)) if TAG == "w50" else ((TAG, WIN),)):
        short = lens < W
        cand = [r for r in rows if r["subset"] == s and r.get("tag") == tag and r["seed"] == 0 and r["model"] in ("gru", "lstm", "tcn", "teacher", "transformer")]
        if not cand or short.sum() == 0:
            pad.setdefault(s, {})[str(W)] = {"n_short": int(short.sum()), "n": int(len(lens))}
            continue
        y = np.minimum(np.asarray(cand[0]["y_test"], float), 125)
        e_all = np.mean([[ (np.asarray(r["test_pred"], float) - y)[i] ** 2 for i in range(len(y))] for r in cand], axis=0)
        pad.setdefault(s, {})[str(W)] = {"n_short": int(short.sum()), "n": int(len(lens)), "rmse_short": float(np.sqrt(e_all[short].mean())), "rmse_rest": float(np.sqrt(e_all[~short].mean())),
                                         "mean_true_rul_short": float(y[short].mean()), "mean_true_rul_rest": float(y[~short].mean())}
OUT["padding"] = pad

# ---------------------------------------------------------------- F naive baselines on window features
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402


def feats(X):
    return np.concatenate([X[:, -1, :], X.mean(1), X[:, -1, :] - X[:, 0, :], X.std(1)], axis=1)


naive = {}
for s in SUBS:
    d = prepare(s, window=WIN, seed=0, condition_aware=s in ("FD002", "FD004"))
    Ftr = np.concatenate([feats(d["X_train"]), feats(d["X_val"])]); ytr = np.concatenate([d["y_train"], d["y_val"]])
    Fte = feats(d["X_test"]); y = d["y_test"]; yc = np.minimum(y, 125)
    out = {}
    const = np.full_like(y, ytr.mean()); out["constant"] = (rmse(const.clip(0, 125), y), rmse(const.clip(0, 125), yc))
    p = Ridge(alpha=1.0).fit(Ftr, ytr).predict(Fte).clip(0, 125); out["ridge"] = (rmse(p, y), rmse(p, yc))
    p = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=0).fit(Ftr, ytr).predict(Fte).clip(0, 125); out["gbm"] = (rmse(p, y), rmse(p, yc))
    naive[s] = {k: {"official": v[0], "capped": v[1]} for k, v in out.items()}
OUT["naive"] = naive

(ROOT / ("results/diagnostics.json" if TAG == "w50" else f"results/diagnostics_{TAG}.json")).write_text(json.dumps(OUT, indent=1))
print(json.dumps({k: OUT[k] for k in ("undertraining", "resolution", "shared_error", "padding", "naive")}, indent=1)[:6000])
