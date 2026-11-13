"""The five accuracy numbers the PDF asks for.

All take plain numpy arrays: y_true (cycles left), y_pred (model's guess).
"""
import numpy as np


def cmapss_score(y_true, y_pred):
    """NASA's asymmetric score. Lower is better, 0 is perfect.

    d = guess - truth
      d < 0 (early guess):  exp(-d/13) - 1   gentler penalty
      d > 0 (late guess):   exp( d/10) - 1   harsher penalty (engine already dead!)
    Summed over engines, so it is NOT an average: it grows with engine count.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def all_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "RMSE": mse ** 0.5,
        "MAE": float(np.mean(np.abs(err))),
        "MSE": mse,
        "R2": 1 - float(np.sum(err ** 2)) / ss_tot,
        "Score": cmapss_score(y_true, y_pred),
    }
