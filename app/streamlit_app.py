"""Streamlit dashboard for the C-MAPSS FD001 LSTM RUL model.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import ARTIFACT_DIR, FIG_DIR, SENSOR_NAMES, Config
from src.data import add_rul, clean, find_data_dir, load_raw
from src.plots import fig_rolling
from src.predict import load_predictor, parse_uploaded_engine

st.set_page_config(page_title="Turbofan RUL Predictor", page_icon="✈️", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_predictor():
    return load_predictor(ARTIFACT_DIR, device="cpu")


@st.cache_data(show_spinner="Loading FD001 data...")
def get_data():
    train, test, rul = load_raw("FD001", find_data_dir("FD001", extra=[ROOT / "data"]))
    return clean(train), clean(test), rul


def health_band(rul: float):
    if rul <= 30:
        return "CRITICAL", "#d62728", "Schedule maintenance immediately."
    if rul <= 70:
        return "WARNING", "#ff9f1c", "Plan maintenance within the next flight window."
    return "HEALTHY", "#2ca02c", "No action needed - keep monitoring."


def gauge(rul: float, cap: int):
    label, color, _ = health_band(rul)
    fig, ax = plt.subplots(figsize=(4.4, 1.0))
    ax.barh([0], [cap], color="#e6e6e6")
    ax.barh([0], [min(rul, cap)], color=color)
    ax.set_xlim(0, cap)
    ax.axis("off")
    ax.set_title(f"{rul:.0f} cycles left  ({label})", color=color, fontsize=11, fontweight="bold")
    return fig


if not (ARTIFACT_DIR / "model.pt").exists():
    st.error("No trained model found. Run `python -m src.train && python -m src.evaluate` first.")
    st.stop()

pred = get_predictor()
train, test, rul_true = get_data()
cap = pred.cap
features = pred.features
metrics_path = ARTIFACT_DIR / "test_metrics.json"
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

st.title("✈️ Turbofan Engine RUL Predictor")
st.caption("NASA C-MAPSS FD001 · 2-layer dropout LSTM (PyTorch) · 30-cycle sliding window · RUL capped at "
           f"{cap}")

tab1, tab2, tab3, tab4 = st.tabs(["🔧 Engine explorer", "📤 Upload engine data",
                                  "📊 Model performance", "ℹ️ About"])

# ------------------------------------------------------------------ Explorer
with tab1:
    c1, c2 = st.columns([1, 3])
    units = sorted(test["unit"].unique())
    unit = c1.selectbox("Test engine", units, index=0)
    g = test[test["unit"] == unit]
    last_cycle = int(g["cycle"].max())
    cut = c1.slider("Observe engine up to cycle", min_value=min(30, last_cycle), max_value=last_cycle,
                    value=last_cycle, help="Simulates the live feed: the model only sees cycles up to this point.")
    seen = g[g["cycle"] <= cut]
    p = pred.predict_last(seen)
    true_at_end = int(rul_true[units.index(unit)])
    true_now = true_at_end + (last_cycle - cut)
    label, color, advice = health_band(p)
    with c1:
        st.metric("Predicted RUL", f"{p:.0f} cycles")
        if cut == last_cycle:
            st.metric("True RUL (ground truth)", f"{true_at_end} cycles", delta=f"{p - true_at_end:+.0f} error",
                      delta_color="off")
        else:
            st.metric("True RUL at this cycle", f"{true_now} cycles", delta=f"{p - true_now:+.0f} error",
                      delta_color="off")
        st.pyplot(gauge(p, cap))
        st.info(advice)
    with c2:
        roll = pred.predict_rolling(seen)
        roll["true_rul"] = np.minimum(true_at_end + (last_cycle - roll["cycle"]), cap)
        st.pyplot(fig_rolling(roll, unit=unit))
        sens = st.multiselect("Sensors to plot", features, default=features[:4],
                              format_func=lambda s: f"{s} - {SENSOR_NAMES.get(s, '')}")
        if sens:
            st.line_chart(seen.set_index("cycle")[sens])

# ------------------------------------------------------------------ Upload
with tab2:
    st.write("Upload a C-MAPSS style file for **one engine** (26 whitespace/comma separated columns: "
             "unit, cycle, 3 operating settings, 21 sensors). At least 1 cycle is required; "
             f"windows shorter than {pred.window} cycles are padded.")
    up = st.file_uploader("Engine history (.txt / .csv)", type=["txt", "csv"])
    use_sample = st.checkbox("Use sample (test engine 24)", value=up is None)
    try:
        if up is not None:
            df = parse_uploaded_engine(up)
        elif use_sample:
            df = test[test["unit"] == 24]
        else:
            df = None
        if df is not None:
            if df["unit"].nunique() > 1:
                u = st.selectbox("File contains several engines - choose one", sorted(df["unit"].unique()))
                df = df[df["unit"] == u]
            p = pred.predict_last(df)
            label, color, advice = health_band(p)
            a, b = st.columns([1, 2])
            a.metric("Predicted RUL", f"{p:.0f} cycles")
            a.pyplot(gauge(p, cap))
            a.info(advice)
            roll = pred.predict_rolling(df)
            b.pyplot(fig_rolling(roll))
            st.download_button("Download rolling predictions (CSV)", roll.to_csv(index=False),
                               "rolling_rul.csv", "text/csv")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not process file: {e}")

# ------------------------------------------------------------------ Performance
with tab3:
    if metrics:
        m = metrics["vs_raw_RUL"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Test RMSE", f"{m['RMSE']:.2f}")
        k2.metric("Test MAE", f"{m['MAE']:.2f}")
        k3.metric("NASA score", f"{m['NASA_score']:.0f}")
        k4.metric("Val RMSE", f"{metrics['val_RMSE']:.2f}")
        st.caption("Metrics vs the official RUL_FD001.txt ground truth (uncapped). "
                   f"Against capped (≤{cap}) truth: RMSE {metrics['vs_capped_RUL(125)']['RMSE']:.2f}, "
                   f"MAE {metrics['vs_capped_RUL(125)']['MAE']:.2f}.")
    pcsv = ARTIFACT_DIR / "test_predictions.csv"
    if pcsv.exists():
        dfp = pd.read_csv(pcsv)
        dfp["error"] = dfp["pred_rul"] - dfp["true_rul"]
        left, right = st.columns(2)
        for name, col in [("06_pred_vs_true", left), ("07_error_hist", right),
                          ("05_training_curves", left), ("09_rolling_example", right)]:
            f = FIG_DIR / f"{name}.png"
            if f.exists():
                col.image(str(f), width='stretch')
        st.dataframe(dfp, width='stretch', height=300)

# ------------------------------------------------------------------ About
with tab4:
    st.markdown(f"""
**Pipeline** — raw FD001 → drop empty column → RUL = last cycle − cycle (capped at {cap}) →
drop constant columns (`{', '.join(pred.meta['constant_cols'])}`) → MinMax scaler fitted on the 80 % training
engines only → 30-cycle windows → 2-layer LSTM (hidden {Config(**pred.meta['config']).hidden_size},
dropout {Config(**pred.meta['config']).dropout}) → RUL.

**Features used ({len(features)}):** `{', '.join(features)}`

**Why cap RUL at {cap}?** Early in life an engine shows no measurable degradation, so a linear target
beyond ~{cap} cycles is unlearnable noise. The piece-wise linear target is the standard fix.

**Health bands:** ≤30 cycles critical · ≤70 warning · otherwise healthy (illustrative thresholds).
""")
