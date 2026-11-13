"""Tiny tests on made-up data (no NASA files needed). Run:  pytest"""
import numpy as np
import pandas as pd

from rul.data import add_train_rul, make_test_windows, make_windows, split_by_engine


def fake_engines(lengths):
    """Engine k has `lengths[k]` cycles; sensor value = engine*1000 + cycle."""
    rows = []
    for eng, n in enumerate(lengths, start=1):
        for c in range(1, n + 1):
            rows.append({"engine": eng, "cycle": c, "x": eng * 1000 + c})
    return pd.DataFrame(rows)


def test_rul_counts_down_and_is_capped():
    df = add_train_rul(fake_engines([200]), cap=125)
    assert df["rul"].iloc[-1] == 0            # last cycle: 0 left
    assert df["rul"].iloc[-2] == 1
    assert df["rul"].iloc[0] == 125           # 199 left -> capped to 125


def test_windows_never_mix_engines():
    df = add_train_rul(fake_engines([40, 35]))
    X, y = make_windows(df, ["x"], window=30)
    assert X.shape == ((40 - 30 + 1) + (35 - 30 + 1), 30, 1)
    engine_of_each_value = X[..., 0] // 1000
    assert (engine_of_each_value == engine_of_each_value[:, :1]).all()


def test_answer_belongs_to_last_row_of_window():
    df = add_train_rul(fake_engines([50]))
    X, y = make_windows(df, ["x"], window=30)
    # first window = cycles 1..30, engine dies after 50 -> 20 left
    assert X[0, -1, 0] == 1030 and y[0] == 20


def test_short_test_engine_is_padded():
    df = fake_engines([10])
    X = make_test_windows(df, ["x"], window=30)
    assert X.shape == (1, 30, 1)
    assert (X[0, :21, 0] == 1001).all()       # first row repeated
    assert X[0, -1, 0] == 1010                # ends at the real last cycle


def test_split_keeps_engines_whole():
    df = fake_engines([40] * 10)
    tr, va = split_by_engine(df, val_frac=0.2, seed=0)
    assert set(tr.engine).isdisjoint(set(va.engine))
    assert va.engine.nunique() == 2


def test_scaler_uses_fit_engines_only():
    """Leakage guard: min/max used for scaling come from the fit engines, not from validation or test."""
    import numpy as np
    from rul.data import add_train_rul, load_raw, prepare
    d = prepare("FD001", seed=0)
    train, _, _ = load_raw("FD001")
    fit_rows = train[train["engine"].isin(d["train_engines"])]
    for s in d["sensors"]:
        assert np.isclose(d["scaler"]["min"][s], fit_rows[s].min()) and np.isclose(d["scaler"]["max"][s], fit_rows[s].max())
    assert set(d["train_engines"]).isdisjoint(set(d["val_engines"]))


def test_testlike_validation_set_is_window_independent_and_in_range():
    """Protocol v2: validation cuts must have true RUL in [1,125] and be identical for every window length."""
    import numpy as np
    from rul.data import prepare
    a = prepare("FD001", window=20, seed=0); b = prepare("FD001", window=80, seed=0)
    assert np.array_equal(a["y_val_tl"], b["y_val_tl"])
    assert a["y_val_tl"].min() >= 1 and a["y_val_tl"].max() <= 125
    assert a["X_val_tl"].shape[1] == 20 and b["X_val_tl"].shape[1] == 80
    assert not (set(a["val_engines"]) & set(a["train_engines"]))          # still engine-grouped
