"""Generate the two notebooks (run once; then execute with nbclient)."""
import nbformat as nbf

def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return n

HEAD = '''import sys, pathlib
ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT))
%matplotlib inline'''

eda = [
("md", "# 01 - Data exploration & preprocessing (C-MAPSS FD001)\n\nFD001: 100 train engines, 100 test engines, **one operating condition**, **one fault mode (HPC degradation)**. Each row = one flight cycle; columns = unit, cycle, 3 operating settings, 21 sensors."),
("code", HEAD + '''
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src.config import Config, SENSOR_NAMES
from src.data import find_data_dir, load_raw, clean, add_rul, select_features, split_units, fit_scaler, apply_scaler, make_windows, make_test_windows
from src import plots
cfg = Config()
print("data folder:", find_data_dir("FD001"))'''),
("md", "## 1. Load & clean\nThe raw files end every line with a space -> pandas creates an **empty trailing column**, which we drop."),
("code", '''train, test, rul_true = load_raw("FD001")
print("raw shapes:", train.shape, test.shape, rul_true.shape)
train, test = clean(train), clean(test)
print("missing values:", int(train.isna().sum().sum()), int(test.isna().sum().sum()))
print("engine life (train): min %d / mean %.0f / max %d cycles" % tuple(train.groupby("unit").size().agg(["min","mean","max"])))
print("test engine length: min %d, max %d" % (test.groupby("unit").size().min(), test.groupby("unit").size().max()))
train.head()'''),
("md", "## 2. RUL target (capped at 125)\nTraining RUL = last cycle of the engine - current cycle. Early in life there is no visible degradation, so we cap at **125** (piece-wise linear target)."),
("code", '''train = add_rul(train, cfg.rul_cap)
plots.fig_rul_labels(train, cfg.rul_cap); plt.show()
train[["unit","cycle","rul_raw","rul"]].tail(3)'''),
("md", "## 3. Constant sensors\nSensors/settings with a single value carry no information and would only add noise (and break scaling)."),
("code", '''features, const = select_features(train, cfg)
print("constant columns :", const)
print("features kept (%d):" % len(features), features)
plots.fig_constant_sensors(train, const); plt.show()'''),
("code", '''plots.fig_sensor_trends(train, features); plt.show()
plots.fig_corr(train, features); plt.show()'''),
("md", "Note `s6` is almost constant (only 21.60/21.61 flicker) - the LSTM learns to ignore it; `op1/op2` are sensor-like noise in FD001, so they are excluded by default (`Config.use_op_settings=False`)."),
("md", "## 4. Engine-level 80/20 split, scaling, windows\nSplit by **engine** (never by row) so no engine appears in both sets. The scaler is fitted on the training engines **only**."),
("code", '''tr_u, va_u = split_units(train["unit"].unique(), cfg.val_frac, cfg.seed)
tr_df, va_df = train[train.unit.isin(tr_u)], train[train.unit.isin(va_u)]
scaler = fit_scaler(tr_df, features)
tr_s, va_s, te_s = (apply_scaler(x, scaler, features) for x in (tr_df, va_df, test))
print("train engines:", len(tr_u), "| val engines:", len(va_u))
X_tr, y_tr, _ = make_windows(tr_s, features, cfg.window)
X_va, y_va, _ = make_windows(va_s, features, cfg.window)
X_te, te_units = make_test_windows(te_s, features, cfg.window)
print("X_train", X_tr.shape, "| X_val", X_va.shape, "| X_test", X_te.shape)
print("scaled train range: %.2f .. %.2f (val may slightly exceed - that is expected, no leakage)" % (tr_s[features].min().min(), tr_s[features].max().max()))
print("val range        : %.2f .. %.2f" % (va_s[features].min().min(), va_s[features].max().max()))'''),
]

train_nb = [
("md", "# 02 - Train LSTM, evaluate on the FD001 test set\n\n2-layer LSTM (hidden 64, dropout 0.3) + small MLP head, Adam, MSE, early stopping on **validation RMSE**, CPU / CUDA / Apple-MPS auto-detected."),
("code", HEAD + '''
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.config import Config, ARTIFACT_DIR
from src.train import train, predict_cycles
from src.evaluate import metrics
from src import plots
cfg = Config()          # edit hyper-parameters here, e.g. Config(hidden_size=128, device="mps")
cfg.to_dict()'''),
("md", "## 1. Train\nSet `RETRAIN=False` to skip training and only load the saved model (training takes ~4 min on one CPU core)."),
("code", '''RETRAIN = True
if RETRAIN:
    out = train(cfg)
    history, d = out["history"], out["data"]
else:
    from src.data import prepare
    history = json.loads((ARTIFACT_DIR / "history.json").read_text()); d = prepare(cfg)
plots.fig_history(history); plt.show()'''),
("md", "## 2. Predict test RUL from each engine's **last 30 cycles**"),
("code", '''from src.predict import load_predictor
pr = load_predictor()
pred = predict_cycles(pr.model, d["X_test"], cfg.rul_cap, pr.device)
true_raw = d["rul_true"].astype(float)
true_cap = np.minimum(true_raw, cfg.rul_cap)
res = pd.DataFrame({"vs official RUL_FD001": metrics(pred, true_raw),
                    "vs capped RUL (<=125)": metrics(pred, true_cap)}).T
res.round(2)'''),
("code", '''plots.fig_pred_vs_true(pred, true_raw); plt.show()
plots.fig_error_hist(pred, true_raw); plt.show()
plots.fig_sorted(pred, true_raw); plt.show()'''),
("md", "## 3. Rolling prediction for one engine\nHow the estimate evolves as more cycles are observed (this is what the Streamlit app shows live)."),
("code", '''u = int(d["test_units"][np.argsort(true_raw)[50]])
g = d["test"][d["test"].unit == u]
roll = pr.predict_rolling(g)
roll["true_rul"] = np.minimum(int(true_raw[list(d["test_units"]).index(u)]) + (g.cycle.max() - roll.cycle), 125)
plots.fig_rolling(roll, unit=u); plt.show()'''),
("md", "## 4. Error analysis\nEngines close to failure matter most; large *positive* errors (late predictions) are the dangerous ones, which is exactly what the NASA score penalises."),
("code", '''df = pd.DataFrame({"unit": d["test_units"], "true": true_raw, "pred": pred.round(1)})
df["error"] = df.pred - df.true
print("RMSE when true RUL <= 30 :", round(float(np.sqrt((df[df.true<=30].error**2).mean())), 2))
print("RMSE when true RUL 30-80 :", round(float(np.sqrt((df[(df.true>30)&(df.true<=80)].error**2).mean())), 2))
print("RMSE when true RUL > 80  :", round(float(np.sqrt((df[df.true>80].error**2).mean())), 2))
df.reindex(df.error.abs().sort_values(ascending=False).index).head(8)'''),
]
nbf.write(nb(eda), "notebooks/01_eda_preprocessing.ipynb")
nbf.write(nb(train_nb), "notebooks/02_train_evaluate.ipynb")
print("notebooks written")
