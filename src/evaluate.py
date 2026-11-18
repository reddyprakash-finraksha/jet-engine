"""Evaluate the saved model on the FD001 test set and create all figures.

Usage:  python -m src.evaluate
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, FIG_DIR, Config
from .data import prepare
from .predict import load_predictor
from .train import predict_cycles


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Asymmetric PHM'08 scoring function (late predictions penalised more)."""
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 10) - 1, np.exp(d / 13) - 1)))


def metrics(pred, true) -> dict:
    return {"RMSE": float(np.sqrt(np.mean((pred - true) ** 2))),
            "MAE": float(np.mean(np.abs(pred - true))),
            "NASA_score": nasa_score(pred, true)}


def evaluate(save_figs: bool = True, verbose: bool = True) -> dict:
    pr = load_predictor()
    cfg = Config(**pr.meta["config"])
    d = prepare(cfg)
    pred = predict_cycles(pr.model, d["X_test"], cfg.rul_cap, pr.device)
    true_raw = d["rul_true"].astype(float)
    true_cap = np.minimum(true_raw, cfg.rul_cap)

    res = {"vs_raw_RUL": metrics(pred, true_raw), "vs_capped_RUL(125)": metrics(pred, true_cap),
           "n_test_engines": int(len(pred)), "val_RMSE": pr.meta["best_val_rmse"]}
    (ARTIFACT_DIR / "test_metrics.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame({"unit": d["test_units"], "true_rul": true_raw.astype(int),
                  "true_rul_capped": true_cap.astype(int),
                  "pred_rul": pred.round(2)}).to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)
    if verbose:
        print(json.dumps(res, indent=2))
    if save_figs:
        from .plots import make_all_figures
        make_all_figures(d, pred, true_raw, true_cap, pr)
    return res


if __name__ == "__main__":
    evaluate()
