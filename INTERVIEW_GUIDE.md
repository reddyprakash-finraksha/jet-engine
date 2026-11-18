# Interview-ready explanation

## 30-second pitch
"I built an end-to-end predictive-maintenance system on NASA's C-MAPSS turbofan data. It predicts how many flights an engine has left from its last 30 cycles of sensor data, using a 2-layer dropout LSTM in PyTorch. On the 100 FD001 test engines it reaches **RMSE 14.7 and MAE 11.1 cycles**, in line with published LSTM baselines. I deployed it as a Streamlit dashboard with live-style rolling predictions and file upload."

## Key decisions (and the "why")
- **RUL cap at 125:** early life shows no degradation, so a linear target there is unlearnable; piece-wise linear target focuses the model on the degradation phase.
- **Drop constant sensors** (op3, s1, s5, s10, s16, s18, s19): zero information; also break/skew scaling. Detected with unique-value check + tolerance because s5=14.62 has float round-off std ≠ 0.
- **Split by engine, not by row:** windows from one engine are highly correlated; row splitting leaks. **Scaler fitted on training engines only** → no leakage.
- **30-cycle windows:** captures the degradation trend while the shortest test engine has 31 cycles; shorter histories are padded.
- **LSTM:** sequence model that learns trend + noise filtering; dropout (0.3) + weight decay + early stopping to limit overfitting on only 80 engines.
- **Target scaled by 125** for stable MSE; metrics reported in cycles.
- **Op settings excluded:** FD001 has one operating condition, so they are noise.

## Metrics
- RMSE/MAE in cycles; **NASA score** is asymmetric (late prediction penalised via exp(d/13), early via exp(−d/10)) — late = engine fails before maintenance.
- I report vs the official uncapped RUL (14.65) and vs capped truth (13.13); say which you use when comparing to papers.

## Likely questions
- **Why does error shrink near failure?** Degradation signal is strongest; far from failure the signal is buried in noise.
- **Why not just train on raw RUL?** Cap explains above.
- **Improvements?** Exponential smoothing of sensors, attention/GRU/1D-CNN or Transformer, ensembling several seeds, asymmetric loss, hyper-parameter search, per-condition normalisation for FD002/FD004, uncertainty (MC-dropout) for maintenance decisions.
- **Is validation RMSE trustworthy?** Only 20 engines → noisy; test set is the honest number; a k-fold over engines would be better.
- **Overfitting evidence?** Train RMSE (~13.9, dropout on) vs val (~13.2): close, no gap.
- **Deployment?** Model + scaler saved in `artifacts/`; `Predictor` class used by both script and Streamlit; CPU inference is milliseconds.
