"""Fast sanity tests:  python -m pytest -q   (or python tests/test_pipeline.py)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from src.config import Config
from src.data import prepare, last_window

def test_pipeline_shapes_and_no_leakage():
    cfg = Config()
    d = prepare(cfg)
    assert d["X_train"].shape[1:] == (30, len(d["features"]))
    assert d["y_train"].max() <= 125 and d["y_val"].max() <= 125
    assert set(d["train_units"]).isdisjoint(d["val_units"])
    assert len(d["train_units"]) == 80 and len(d["val_units"]) == 20
    assert {"op3", "s1", "s5", "s10", "s16", "s18", "s19"} <= set(d["constant_cols"])
    assert d["X_test"].shape[0] == len(d["rul_true"]) == 100
    # scaler fitted on train split only -> train range exactly [0,1]
    tr = d["scaler"].transform(d["train"][d["train"].unit.isin(d["train_units"])][d["features"]].to_numpy())
    assert np.isclose(tr.min(), 0) and np.isclose(tr.max(), 1)

def test_last_window_padding():
    a = np.arange(10).reshape(5, 2).astype(float)
    w = last_window(a, 8)
    assert w.shape == (8, 2) and (w[0] == a[0]).all() and (w[-1] == a[-1]).all()

if __name__ == "__main__":
    test_pipeline_shapes_and_no_leakage(); test_last_window_padding(); print("all tests passed")
