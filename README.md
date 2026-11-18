# ✈️ Turbofan Remaining Useful Life (RUL) Prediction — NASA C-MAPSS FD001

End-to-end deep learning project: raw NASA run-to-failure sensor data → 2-layer dropout **LSTM (PyTorch)** → RUL prediction → **Streamlit** dashboard.

| Test result (100 engines, last 30 cycles) | RMSE | MAE | NASA score |
|---|---|---|---|
| vs official `RUL_FD001.txt` | **14.65** | **11.14** | 362 |
| vs RUL capped at 125 | 13.13 | 10.07 | 261 |

Best validation RMSE: 13.23 (20 held-out engines). Training: 57 epochs, early stopped, ~4 min on 1 CPU core.

## Pipeline
1. Locate `train_/test_/RUL_FD001.txt` in `data/`; drop the empty trailing column (lines end with a space).
2. RUL = last cycle − current cycle, **capped at 125**.
3. Detect constant columns (`op3, s1, s5, s10, s16, s18, s19`) → 15 sensors remain (op settings off by default: single operating condition → noise).
4. **80/20 split by engine**; MinMax scaler fitted on the training engines only.
5. 30-cycle stride-1 sliding windows (14,459 train / 3,272 val).
6. LSTM(15→64, 2 layers, dropout 0.3) → Dropout → Linear(32) → ReLU → Linear(1). Adam, MSE, grad-clip, ReduceLROnPlateau, early stopping (patience 12) on validation RMSE. Device: CUDA / Apple MPS / CPU auto.
7. Test: each engine's last 30 cycles → RUL; RMSE, MAE, NASA asymmetric score.

## Structure
```
data/        train_FD001.txt test_FD001.txt RUL_FD001.txt
src/         config.py data.py model.py train.py evaluate.py predict.py plots.py
app/         streamlit_app.py
notebooks/   01_eda_preprocessing.ipynb  02_train_evaluate.ipynb  (executed, with outputs)
artifacts/   model.pt scaler.joblib metadata.json history.json test_metrics.json test_predictions.csv figures/
tests/       test_pipeline.py  test_app.py
```

## Run
```
python -m venv .venv && .venv\Scripts\activate        # Windows  (Mac: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q tests/test_pipeline.py
python -m src.train --device auto                      # Apple Silicon: --device mps
python -m src.evaluate                                 # metrics + figures
streamlit run app/streamlit_app.py
```
Or everything at once: `bash run_all.sh`. Hyper-parameters: `python -m src.train --hidden-size 128 --window 40 --epochs 150`.

## Deploy on Streamlit Community Cloud (free)
1. Push the whole folder (**including `artifacts/` and `data/`**) to a GitHub repo.
2. share.streamlit.io → New app → pick repo, branch `main`, main file `app/streamlit_app.py`.
3. Advanced settings → Python 3.11 or 3.12 → Deploy. `requirements.txt` already pulls CPU-only torch.

## Dashboard
- **Engine explorer:** pick a test engine, slide "observe up to cycle" to simulate live monitoring; predicted vs true RUL, health gauge, rolling prediction curve, sensor plots.
- **Upload:** upload one engine's 26-column file → RUL + downloadable rolling predictions.
- **Performance:** metrics, figures, per-engine table.

## Limitations
Single operating condition/fault mode (FD001); simulated data; 15–30 cycle error is typical. FD002–FD004 need per-operating-condition normalisation. Health-band thresholds (30/70) are illustrative.

Reference: Saxena et al., *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM'08.
