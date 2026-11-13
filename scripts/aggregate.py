"""Load every results/suite/*.json into one flat DataFrame; mean/std over seeds. Used by build_report.py."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

SUITE = Path(__file__).resolve().parents[1] / "results" / "suite"


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(SUITE / "*.json"))):
        d = json.load(open(f))
        r = {k: d.get(k) for k in ("kind", "subset", "model", "tag", "seed", "teacher", "student", "params",
                                   "macs", "acs", "energy_pj", "epochs_run", "train_seconds",
                                   "test_spike_rate", "rate1", "rate2", "window", "select")}
        for split in ("val", "test", "val_pseudo", "val_tl"):
            for m, v in (d.get(split) or {}).items():
                r[f"{split}_{m}"] = v
        if d.get("test_pred") and d.get("y_test"):   # secondary convention: labels capped at 125 like the training target
            y = np.minimum(np.asarray(d["y_test"], float), 125.0); q = np.asarray(d["test_pred"], float)
            r["test_RMSE_c125"] = float(np.sqrt(np.mean((q - y) ** 2)))
            r["test_MAE_c125"] = float(np.mean(np.abs(q - y)))
        r["id"] = d.get("id")
        rows.append(r)
    df = pd.DataFrame(rows)
    df["tag"] = df["tag"].fillna("")
    return df


def agg(df: pd.DataFrame, by, cols=("test_RMSE", "test_Score", "test_MAE", "test_R2", "energy_pj", "params", "test_spike_rate")):
    """mean/std/n per group; std is the sample std over seeds (NaN when n==1)."""
    cols = [c for c in cols if c in df.columns]
    g = df.groupby(by, dropna=False)[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).join(g.size().rename("n"))
    return out.reset_index()


if __name__ == "__main__":
    df = load()
    print(df.shape)
    print(agg(df[df.kind == "base"], ["subset", "model", "tag"])[["subset", "model", "tag", "test_RMSE_mean", "test_RMSE_std", "n"]].to_string())
