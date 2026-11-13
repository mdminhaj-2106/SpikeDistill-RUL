"""Sanity check: load FD001, print every shape, save one EDA plot.

Run from the project root:   python scripts/check_data.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

from rul.data import add_train_rul, load_raw, prepare

d = prepare("FD001")

print("useful sensors:", d["sensors"], f"({len(d['sensors'])})")
for k in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
    print(f"{k:8s} {d[k].shape}")
print("y_train range:", d["y_train"].min(), "to", d["y_train"].max(), "(should be 0 to 125)")
print("X_train range:", d["X_train"].min().round(3), "to", d["X_train"].max().round(3), "(about 0 to 1)")

# The "green banana" plot: one sensor over an engine's whole life.
train, _, _ = load_raw("FD001")
train = add_train_rul(train, cap=10_000)  # uncapped, just for the plot
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=False)
for ax, sensor in zip(axes.flat, ["s2", "s7", "s11", "s15"]):
    for eng in [1, 2, 3, 4, 5]:
        e = train[train.engine == eng]
        ax.plot(e["rul"], e[sensor], lw=0.8, label=f"engine {eng}")
    ax.invert_xaxis()  # left = far from failure, right = failure
    ax.set_title(sensor)
    ax.set_xlabel("cycles left (true)")
axes.flat[0].legend(fontsize=7)
fig.tight_layout()

out = Path("results/figures/sensors_vs_cycles_left.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120)
print("saved", out)
