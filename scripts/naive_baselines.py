"""Hand-made-feature baselines on the SAME split as the networks (train engines only; validation engines used only for ridge alpha and GBM early stopping).
Features per window: last value, window mean, last-minus-first, window std, per sensor. Models: constant (train mean), ridge, histogram gradient boosting.
5 seeds (0-4), window = protocol-v2 window, regime-standardised for FD002/FD004. Output: results/naive_v2.json (capped and official RMSE, mean/sd)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from rul.data import prepare
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

W = int(json.load(open(ROOT / "results/window_choice_v2.json"))["window"])
feats = lambda X: np.concatenate([X[:, -1], X.mean(1), X[:, -1] - X[:, 0], X.std(1)], axis=1)
rm = lambda p, y: float(np.sqrt(np.mean((np.clip(p, 0, 125) - y) ** 2)))
out = {}
for s in ["FD001", "FD002", "FD003", "FD004"]:
    rows = {"constant": [], "ridge": [], "gbm": []}
    for seed in range(5):
        d = prepare(s, window=W, seed=seed, condition_aware=s in ("FD002", "FD004"))
        Ftr, Fva, Fte = feats(d["X_train"]), feats(d["X_val_tl"]), feats(d["X_test"]); ytr, yva = d["y_train"], d["y_val_tl"]
        y, yc = d["y_test"], np.minimum(d["y_test"], 125)
        rows["constant"].append((rm(np.full_like(y, ytr.mean()), y), rm(np.full_like(y, ytr.mean()), yc)))
        best = min((rm(Ridge(alpha=a).fit(Ftr, ytr).predict(Fva), yva), a) for a in (0.1, 1, 10, 100, 1000))[1]
        p = Ridge(alpha=best).fit(Ftr, ytr).predict(Fte); rows["ridge"].append((rm(p, y), rm(p, yc)))
        g = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=seed).fit(Ftr, ytr)
        p = g.predict(Fte); rows["gbm"].append((rm(p, y), rm(p, yc)))
    out[s] = {k: {"official": [float(np.mean([a for a, _ in v])), float(np.std([a for a, _ in v], ddof=1))],
                  "capped": [float(np.mean([b for _, b in v])), float(np.std([b for _, b in v], ddof=1))], "n": len(v)} for k, v in rows.items()}
    print(s, {k: round(v["capped"][0], 2) for k, v in out[s].items()}, flush=True)
(ROOT / "results/naive_v2.json").write_text(json.dumps(out, indent=1))
