"""Protocol-v2 results -> results/analysis_v2.json (one source of numbers for poster, report and deck).

  main        per subset and model (tag v2): params, val_tl RMSE, test RMSE official + capped-125, MAE, R2, score, seconds (mean, sd, n over seeds)
  ties        seed-ensembled test predictions per model; paired bootstrap over test ENGINES vs the best model -> which models are statistically tied
  distill     teacher vs no teacher (tag v2h), per subset and student, paired by seed: mean, sd, 95% CI, t-test, Wilcoxon, effect size, seeds improved
  pooled      one model trained on all four subsets vs the same model trained per subset (seeds 0-4), paired t-test per subset
  energy      operation counts and energy of the headline students at the chosen window
  window      the sweep table from choose_window_v2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from aggregate import load  # noqa: E402

df = load(); df["tag"] = df["tag"].fillna(""); df["teacher"] = df["teacher"].fillna("none")
SUBS = ["FD001", "FD002", "FD003", "FD004"]
rng = np.random.default_rng(0)


def ms(x):
    x = pd.Series(x).dropna()
    return None if len(x) == 0 else {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "n": int(len(x))}


def rows(sub, model, tag="v2", kind="base"):
    return df[(df.kind == kind) & (df.subset == sub) & (df.model == model) & (df.tag == tag)]


out = {"main": {}, "ties": {}, "distill": {}, "pooled": {}, "energy": {}, "window": None}
wc = ROOT / "results/window_choice_v2.json"
if wc.exists():
    out["window"] = json.load(open(wc))

# ---------------------------------------------------------------- main table
models = sorted(df[(df.kind == "base") & (df.tag == "v2")].model.unique())
for s in SUBS:
    out["main"][s] = {}
    for m in models:
        x = rows(s, m)
        if len(x):
            out["main"][s][m] = {"params": ms(x.params), "val_tl": ms(x.val_tl_RMSE), "rmse": ms(x.test_RMSE), "rmse_c125": ms(x.test_RMSE_c125), "mae": ms(x.test_MAE),
                                 "r2": ms(x.test_R2), "score": ms(x.test_Score), "seconds": ms(x.train_seconds), "macs": ms(x.macs), "energy_nj": ms(x.energy_pj / 1000)}

# ---------------------------------------------------------------- which models are tied with the best (seed-ensembled, engine bootstrap)
for s in SUBS:
    P, y = {}, None
    for m in models:
        preds = []
        for f in (ROOT / "results/suite").glob(f"base__{s}__{m}__v2__*.json"):
            r = json.load(open(f))
            if r["seed"] >= 5:                       # same five seeds for every model (the Mamba family only has 0-4): equal ensemble size
                continue
            preds.append(np.asarray(r["test_pred"], float)); y = np.minimum(np.asarray(r["y_test"], float), 125)
        if len(preds) >= 3:
            P[m] = np.mean(preds, axis=0)
    if len(P) < 3:
        continue
    n = len(y); e2 = {m: (p - y) ** 2 for m, p in P.items()}
    rm = {m: float(np.sqrt(v.mean())) for m, v in e2.items()}; best = min(rm, key=rm.get)
    idx = rng.integers(0, n, (2000, n))
    res = {}
    for m in P:
        d = np.sqrt(e2[m][idx].mean(1)) - np.sqrt(e2[best][idx].mean(1))
        lo, hi = np.percentile(d, [2.5, 97.5])
        res[m] = {"rmse_c125_seed_ensemble": rm[m], "diff_to_best": rm[m] - rm[best], "ci95": [float(lo), float(hi)], "tied_with_best": bool(lo <= 0)}
    out["ties"][s] = {"best": best, "n_engines": n, "models": res}

# ---------------------------------------------------------------- distillation, paired by seed
for s in SUBS:
    for stu in ("spike", "smallgru", "smalllstm"):
        a = df[(df.kind == "distill") & (df.tag == "v2h") & (df.subset == s) & (df.student == stu) & (df.teacher == "none")].set_index("seed")
        b = df[(df.kind == "distill") & (df.tag == "v2h") & (df.subset == s) & (df.student == stu) & (df.teacher == "teacher")].set_index("seed")
        sd = sorted(set(a.index) & set(b.index))
        if len(sd) < 3:
            continue
        rec = {"seeds": sd, "none": ms(a.loc[sd, "test_RMSE_c125"]), "teacher": ms(b.loc[sd, "test_RMSE_c125"]), "params": ms(b.loc[sd, "params"]),
               "spike_rate": ms(b.loc[sd, "test_spike_rate"]) if stu == "spike" else None}
        for key, col in (("c125", "test_RMSE_c125"), ("raw", "test_RMSE")):
            d = (b.loc[sd, col] - a.loc[sd, col]).to_numpy(float)
            t, p = stats.ttest_rel(b.loc[sd, col], a.loc[sd, col]); h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            rec[key] = {"diffs": d.tolist(), "mean": float(d.mean()), "sd": float(d.std(ddof=1)), "ci95": [float(d.mean() - h), float(d.mean() + h)], "t": float(t), "p": float(p),
                        "dz": float(d.mean() / d.std(ddof=1)), "improved": int((d < 0).sum()), "n": len(d),
                        "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else None}
        out["distill"].setdefault(s, {})[stu] = rec

# ---------------------------------------------------------------- pooled vs per-subset training
pf = [json.load(open(f)) for f in sorted((ROOT / "results/suite").glob("pooled__ALL__*__v2__*.json"))]
for m in sorted({r["model"] for r in pf}):
    out["pooled"][m] = {}
    for s in SUBS:
        pooled = {r["seed"]: r["test_by_subset"][s]["RMSE_c125"] for r in pf if r["model"] == m}
        sep = rows(s, m).set_index("seed")["test_RMSE_c125"].to_dict()
        sd = sorted(set(pooled) & set(sep))
        if len(sd) >= 3:
            a = np.array([sep[k] for k in sd]); b = np.array([pooled[k] for k in sd]); d = b - a
            t, p = stats.ttest_rel(b, a)
            out["pooled"][m][s] = {"separate": ms(a), "pooled": ms(b), "diff_mean": float(d.mean()), "diff_sd": float(d.std(ddof=1)), "t": float(t), "p": float(p), "n": len(d), "improved": int((d < 0).sum())}

# ---------------------------------------------------------------- energy of the headline students
for s in SUBS:
    for stu, tch in (("spike", "teacher"), ("spike", "none"), ("smallgru", "none"), ("smallgru", "teacher")):
        x = df[(df.kind == "distill") & (df.tag == "v2h") & (df.subset == s) & (df.student == stu) & (df.teacher == tch)]
        if len(x):
            out["energy"].setdefault(s, {})[f"{stu}/{tch}"] = {"macs": ms(x.macs), "acs": ms(x.acs), "energy_nj": ms(x.energy_pj / 1000), "rate": ms(x.test_spike_rate), "params": ms(x.params)}

(ROOT / "results/analysis_v2.json").write_text(json.dumps(out, default=float))
print({k: (len(v) if hasattr(v, "__len__") else v) for k, v in out.items()})
for s, t in out["ties"].items():
    tied = [m for m, r in t["models"].items() if r["tied_with_best"]]
    print(s, "best", t["best"], f"{t['models'][t['best']]['rmse_c125_seed_ensemble']:.2f}", "| tied with best:", len(tied), "of", len(t["models"]))
for s, by in out["distill"].items():
    for stu, r in by.items():
        print(s, stu, f"teacher effect (capped) {r['c125']['mean']:+.2f} p={r['c125']['p']:.3f} improved {r['c125']['improved']}/{r['c125']['n']}")
for m, by in out["pooled"].items():
    print("pooled", m, {s: f"{r['diff_mean']:+.2f}(p={r['p']:.2f})" for s, r in by.items()})
