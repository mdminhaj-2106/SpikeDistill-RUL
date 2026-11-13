"""Load NASA C-MAPSS and turn it into (window, answer) arrays.

Zero-Leakage Implementation:
  1. load_raw                  Read raw text files
  2. add_train_rul             Add cycles left column (capped at cap)
  3. split_by_engine           Split whole engines into fit and validation folds FIRST
  4. find_useful_sensors       Drop constant/near-constant sensors on FIT split ONLY
  5. fit_scaler / scale        Min-Max or condition-aware scaling on FIT split ONLY
  6. make_windows              Slide 30-cycle windows within each engine
  7. make_test_windows         Extract final 30 cycles per test engine (with padding)
  8. make_pseudo_test_windows  Extract truncated test-like windows from validation engines
  prepare()                    Orchestrates the entire leakage-free pipeline
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SETTINGS = ["setting1", "setting2", "setting3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["engine", "cycle"] + SETTINGS + SENSORS


# 1 ----------------------------------------------------------------------------
def load_raw(subset: str = "FD001", data_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Returns (train_df, test_df, test_true_rul).

    test_true_rul[i] = true cycles left for test engine i+1 at its LAST row.
    """
    d = Path(data_dir)
    # float64 everywhere: some sensors parse as int64 and cannot hold scaled values
    read = lambda name: pd.read_csv(d / name, sep=r"\s+", header=None, names=COLUMNS, dtype={c: "float64" for c in COLUMNS[2:]})
    train = read(f"train_{subset}.txt")
    test = read(f"test_{subset}.txt")
    true_rul = pd.read_csv(d / f"RUL_{subset}.txt", sep=r"\s+", header=None)[0].to_numpy()
    return train, test, true_rul


# 2 ----------------------------------------------------------------------------
def add_train_rul(train: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """Cycles left = last cycle of that engine - current cycle, capped at cap."""
    train = train.copy()
    last = train.groupby("engine")["cycle"].transform("max")
    train["rul"] = (last - train["cycle"]).clip(upper=cap)
    train["rul_uncapped"] = last - train["cycle"]
    return train


# 3 ----------------------------------------------------------------------------
def split_by_engine(train: pd.DataFrame, val_frac: float = 0.2, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly pick whole engines for validation (never split one engine)."""
    engines = train["engine"].unique()
    rng = np.random.default_rng(seed)
    n_val = int(round(len(engines) * val_frac))
    val_engines = set(rng.choice(engines, size=n_val, replace=False))
    is_val = train["engine"].isin(val_engines)
    return train[~is_val].copy(), train[is_val].copy()


# 4 ----------------------------------------------------------------------------
def find_useful_sensors(train_split: pd.DataFrame, min_std: float = 0.01) -> List[str]:
    """Keep sensors whose spread (std) is above min_std on the FIT split.

    In FD001/FD003 this drops s1, s5, s6, s10, s16, s18, s19 and keeps 14.
    """
    stds = train_split[SENSORS].std()
    return [s for s in SENSORS if stds[s] > min_std]


# 5 ----------------------------------------------------------------------------
def fit_scaler(df: pd.DataFrame, cols: List[str]) -> Dict[str, pd.Series]:
    """Learn min and max of each column from FIT rows only."""
    return {"min": df[cols].min(), "max": df[cols].max()}


def scale(df: pd.DataFrame, cols: List[str], scaler: Dict[str, pd.Series]) -> pd.DataFrame:
    """(value - min) / (max - min), using statistics learned strictly from train."""
    df = df.copy()
    span = (scaler["max"] - scaler["min"]).replace(0, 1.0)
    df[cols] = (df[cols] - scaler["min"]) / span
    return df


# 5b. Multi-Regime Scaling for FD002 & FD004 ------------------------------------
def fit_regime_scaler(
    df: pd.DataFrame, cols: List[str], settings_cols: List[str] = SETTINGS, n_regimes: int = 6
) -> Dict:
    """Fit K-Means clustering on settings and compute regime-specific statistics.

    Fitted strictly on training split engines!
    """
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    regimes = kmeans.fit_predict(df[settings_cols])
    df_temp = df[cols].copy()
    df_temp["_regime"] = regimes

    regime_stats = {}
    for r in range(n_regimes):
        subset = df_temp[df_temp["_regime"] == r][cols]
        if len(subset) == 0:
            regime_stats[r] = {"mean": df[cols].mean(), "std": df[cols].std().replace(0, 1.0)}
        else:
            regime_stats[r] = {
                "mean": subset.mean(),
                "std": subset.std().replace(0, 1.0).fillna(1.0),
            }

    global_stats = {"mean": df[cols].mean(), "std": df[cols].std().replace(0, 1.0)}
    return {"kmeans": kmeans, "regime_stats": regime_stats, "global_stats": global_stats}


def scale_by_regime(
    df: pd.DataFrame, cols: List[str], regime_scaler: Dict, settings_cols: List[str] = SETTINGS
) -> pd.DataFrame:
    """Standardize sensor values per operational regime learned from train."""
    df = df.copy()
    regimes = regime_scaler["kmeans"].predict(df[settings_cols])
    scaled_blocks = []
    for r in range(len(regime_scaler["regime_stats"])):
        mask = regimes == r
        if mask.any():
            block = df.loc[mask, cols].copy()
            m = regime_scaler["regime_stats"][r]["mean"]
            s = regime_scaler["regime_stats"][r]["std"]
            df.loc[mask, cols] = (block - m) / s
    return df


# 5c. Causal smoothing (feature engineering) -----------------------------------
def smooth_causal(df: pd.DataFrame, cols: List[str], alpha: float) -> pd.DataFrame:
    """Exponential moving average within each engine, past-only (adjust=False).

    Row t only sees rows <= t of the SAME engine, so it can never leak the future or other engines.
    """
    df = df.copy()
    df[cols] = df.groupby("engine")[cols].transform(lambda c: c.ewm(alpha=alpha, adjust=False).mean())
    return df


# 6 ----------------------------------------------------------------------------
def make_windows(df: pd.DataFrame, cols: List[str], window: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """For each engine separately: sliding windows of `window` rows, answer = rul at last row.

    Returns X: (n_windows, window, n_sensors), y: (n_windows,)
    """
    X, y = [], []
    for _, e in df.groupby("engine"):
        vals = e[cols].to_numpy(dtype=np.float32)
        rul = e["rul"].to_numpy(dtype=np.float32)
        for end in range(window, len(e) + 1):
            X.append(vals[end - window:end])
            y.append(rul[end - 1])
    return np.stack(X), np.array(y, dtype=np.float32)


# 7 ----------------------------------------------------------------------------
def make_test_windows(test: pd.DataFrame, cols: List[str], window: int = 30) -> np.ndarray:
    """One window per test engine: its last `window` rows.

    Engines shorter than `window` get their first row repeated at the start.
    """
    X = []
    for _, e in test.groupby("engine"):
        vals = e[cols].to_numpy(dtype=np.float32)
        if len(vals) < window:
            pad = np.repeat(vals[:1], window - len(vals), axis=0)
            vals = np.concatenate([pad, vals])
        X.append(vals[-window:])
    return np.stack(X)


# 8. Pseudo-Test Windows for Unbiased Validation --------------------------------
def make_val_pseudo_test_windows(
    va_df: pd.DataFrame, cols: List[str], window: int = 30, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts exactly one truncated test-like window per validation engine.

    Truncation cycle is picked randomly in [30, L_end - 5] to emulate test conditions
    without touching test labels.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _, e in va_df.groupby("engine"):
        L = len(e)
        if L <= window:
            trunc = L
        else:
            trunc = rng.integers(window, max(window + 1, L - 5))
        vals = e[cols].iloc[:trunc].to_numpy(dtype=np.float32)
        if len(vals) < window:
            pad = np.repeat(vals[:1], window - len(vals), axis=0)
            vals = np.concatenate([pad, vals])
        X.append(vals[-window:])
        # true uncapped cycles remaining from truncation point
        y.append(float(L - trunc))
    return np.stack(X), np.array(y, dtype=np.float32)


# All together -----------------------------------------------------------------
def prepare(
    subset: str = "FD001",
    data_dir: str = "data/raw",
    window: int = 30,
    cap: int = 125,
    val_frac: float = 0.2,
    seed: int = 0,
    condition_aware: bool = False,
    smooth_alpha: Optional[float] = None,
    sensors: Optional[List[str]] = None,
) -> Dict:
    """Orchestrates loading, leakage-free splitting, sensor selection, scaling, and windowing."""
    train, test, true_rul = load_raw(subset, data_dir)
    train = add_train_rul(train, cap)

    # 1. SPLIT FIRST to guarantee ZERO leakage
    tr, va = split_by_engine(train, val_frac, seed)

    # 2. SELECT SENSORS on FIT SPLIT ONLY
    sensors = list(sensors) if sensors else find_useful_sensors(tr)

    # 3. FIT SCALER on FIT SPLIT ONLY
    if condition_aware and subset in ("FD002", "FD004"):
        scaler = fit_regime_scaler(tr, sensors, SETTINGS)
        tr = scale_by_regime(tr, sensors, scaler, SETTINGS)
        va = scale_by_regime(va, sensors, scaler, SETTINGS)
        te = scale_by_regime(test, sensors, scaler, SETTINGS)
    else:
        scaler = fit_scaler(tr, sensors)
        tr = scale(tr, sensors, scaler)
        va = scale(va, sensors, scaler)
        te = scale(test, sensors, scaler)

    # 3b. OPTIONAL causal EMA smoothing (per engine, past-only)
    if smooth_alpha:
        tr, va, te = (smooth_causal(d, sensors, smooth_alpha) for d in (tr, va, te))

    # 4. CONSTRUCT WINDOWS
    X_train, y_train = make_windows(tr, sensors, window)
    X_val, y_val = make_windows(va, sensors, window)
    X_val_pseudo, y_val_pseudo = make_val_pseudo_test_windows(va, sensors, window, seed=seed)
    X_val_tl, y_val_tl = make_val_testlike_windows(va, sensors, window, cap)
    X_test = make_test_windows(te, sensors, window)
    y_test = true_rul.astype(np.float32)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_val_pseudo": X_val_pseudo,
        "y_val_pseudo": y_val_pseudo,
        "X_val_tl": X_val_tl,
        "y_val_tl": y_val_tl,
        "X_test": X_test,
        "y_test": y_test,
        "sensors": sensors,
        "scaler": scaler,
        "train_engines": tr["engine"].unique(),
        "val_engines": va["engine"].unique(),
    }


# 9. Test-like validation windows (protocol v2) ---------------------------------
def make_val_testlike_windows(va_df: pd.DataFrame, cols: List[str], window: int = 30, cap: int = 125) -> Tuple[np.ndarray, np.ndarray]:
    """Every cut of every validation engine whose true remaining life is in [1, cap], windowed exactly like the test set.

    A test engine is cut at some cycle t and scored on its last `window` rows (first row repeated if t < window), against the
    RUL that remains after t. Here: for each validation engine of length L, t runs over L-cap .. L-1, so the label L-t lies in
    [1, cap] (the same range the capped test labels span) and no cap is needed on the label.
    The set depends only on the engine split, NOT on `window`, so it is identical for every window length (fair comparison).
    """
    X, y = [], []
    for _, e in va_df.groupby("engine"):
        vals_all = e[cols].to_numpy(dtype=np.float32)
        L = len(e)
        for t in range(max(1, L - cap), L):
            vals = vals_all[:t]
            if len(vals) < window:
                vals = np.concatenate([np.repeat(vals[:1], window - len(vals), axis=0), vals])
            X.append(vals[-window:])
            y.append(float(L - t))
    return np.stack(X), np.array(y, dtype=np.float32)
