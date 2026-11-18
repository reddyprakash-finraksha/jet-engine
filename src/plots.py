"""All visualisations used in the README / notebooks / app."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ARTIFACT_DIR, FIG_DIR, SENSOR_NAMES

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig_rul_labels(train, cap=125):
    unit = train["unit"].iloc[0]
    g = train[train["unit"] == unit]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(g["cycle"], g["rul_raw"], "--", label="raw linear RUL")
    ax.plot(g["cycle"], g["rul"], lw=2.2, label=f"capped RUL (cap={cap})")
    ax.set(xlabel="cycle", ylabel="RUL (cycles)", title=f"RUL target construction - engine {unit}")
    ax.legend()
    return fig


def fig_sensor_trends(train, sensors, units=(1, 2, 3, 4, 5)):
    n = len(sensors)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.6 * rows), sharex=False)
    for ax, s in zip(np.ravel(axes), sensors):
        for u in units:
            g = train[train["unit"] == u]
            ax.plot(g["cycle"], g[s], lw=0.9)
        ax.set_title(SENSOR_NAMES.get(s, s), fontsize=9)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle("Informative sensors over engine life (5 engines)", y=1.0)
    fig.tight_layout()
    return fig


def fig_constant_sensors(train, const_cols):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    stds = train[const_cols].std()
    ax.bar(stds.index, stds.values)
    ax.set(title="Constant columns detected (std = 0) and dropped", ylabel="std")
    return fig


def fig_corr(train, features):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    c = train[features + ["rul"]].corr()
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(c)), c.columns, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Feature / RUL correlation")
    return fig


def fig_history(history):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(history["train_rmse"], label="train RMSE (dropout on)")
    ax.plot(history["val_rmse"], label="validation RMSE")
    best = int(np.argmin(history["val_rmse"]))
    ax.axvline(best, color="grey", ls=":", label=f"best epoch {best + 1}")
    ax.set(xlabel="epoch", ylabel="RMSE (cycles)", title="Training curves")
    ax.legend()
    return fig


def fig_pred_vs_true(pred, true, title="Predicted vs true RUL (test)"):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(true, pred, s=22, alpha=0.75)
    lim = [0, max(true.max(), pred.max()) + 5]
    ax.plot(lim, lim, "r--", lw=1, label="perfect")
    ax.set(xlabel="true RUL", ylabel="predicted RUL", title=title, xlim=lim, ylim=lim)
    ax.legend()
    return fig


def fig_error_hist(pred, true):
    err = pred - true
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(err, bins=25, edgecolor="white")
    ax.axvline(0, color="r", ls="--")
    ax.set(xlabel="prediction error (pred - true)  [>0 = late/dangerous]", ylabel="engines",
           title=f"Error distribution (mean {err.mean():+.1f}, std {err.std():.1f})")
    return fig


def fig_sorted(pred, true):
    o = np.argsort(true)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(true[o], "o-", ms=3, label="true RUL")
    ax.plot(pred[o], "s-", ms=3, label="predicted RUL")
    ax.set(xlabel="test engines sorted by true RUL", ylabel="RUL (cycles)",
           title="Test predictions, engines sorted by true RUL")
    ax.legend()
    return fig


def fig_rolling(roll_df, true_rul_at_end=None, unit=None):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(roll_df["cycle"], roll_df["pred_rul"], lw=2, label="predicted RUL")
    if "true_rul" in roll_df:
        ax.plot(roll_df["cycle"], roll_df["true_rul"], "--", label="true RUL (capped)")
    ax.set(xlabel="cycle", ylabel="RUL (cycles)",
           title=f"Rolling RUL prediction{'' if unit is None else f' - engine {unit}'}")
    ax.legend()
    return fig


def make_all_figures(d, pred, true_raw, true_cap, predictor):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hist = json.loads((ARTIFACT_DIR / "history.json").read_text())
    train = d["train"]
    figs = {
        "01_rul_labels": fig_rul_labels(train),
        "02_constant_sensors": fig_constant_sensors(train, d["constant_cols"]),
        "03_sensor_trends": fig_sensor_trends(train, d["features"]),
        "04_correlation": fig_corr(train, d["features"]),
        "05_training_curves": fig_history(hist),
        "06_pred_vs_true": fig_pred_vs_true(pred, true_raw),
        "07_error_hist": fig_error_hist(pred, true_raw),
        "08_sorted_predictions": fig_sorted(pred, true_raw),
    }
    # rolling example: pick the test engine whose true RUL is median
    units = d["test_units"]
    u = int(units[np.argsort(true_raw)[len(units) // 2]])
    g = d["test"][d["test"]["unit"] == u]
    roll = predictor.predict_rolling(g)
    last = int(g["cycle"].max())
    roll["true_rul"] = np.minimum(int(true_raw[list(units).index(u)]) + (last - roll["cycle"]), 125)
    figs["09_rolling_example"] = fig_rolling(roll, unit=u)
    for name, f in figs.items():
        f.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
        plt.close(f)
