"""Data loading, cleaning, RUL labelling, scaling and sliding-window creation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import ALL_COLS, DATA_DIR, OP_COLS, ROOT, SENSOR_COLS, Config


# --------------------------------------------------------------------------- #
# Locating + reading files
# --------------------------------------------------------------------------- #
def find_data_dir(subset: str = "FD001", extra: Iterable[Path] = ()) -> Path:
    """Return the folder that contains train_/test_/RUL_<subset>.txt."""
    needed = [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"]
    candidates = [*extra, DATA_DIR, ROOT / "data", Path.cwd() / "data", Path.cwd(),
                  ROOT, ROOT.parent / "data", Path("/mnt/user-data/uploads")]
    for d in candidates:
        d = Path(d)
        if all((d / n).exists() for n in needed):
            return d
    raise FileNotFoundError(
        f"Could not find {needed}. Put them in {DATA_DIR} (the 'data' folder).")


def read_cmapss(path: Path | str) -> pd.DataFrame:
    """Read a raw C-MAPSS text file (whitespace separated, trailing blank column)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")          # trailing separator -> empty column
    if df.shape[1] != len(ALL_COLS):
        raise ValueError(f"{path}: expected {len(ALL_COLS)} columns, got {df.shape[1]}")
    df.columns = ALL_COLS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_raw(subset: str = "FD001", data_dir: Path | str | None = None):
    d = Path(data_dir) if data_dir else find_data_dir(subset)
    train = read_cmapss(d / f"train_{subset}.txt")
    test = read_cmapss(d / f"test_{subset}.txt")
    rul = pd.read_csv(d / f"RUL_{subset}.txt", sep=r"\s+", header=None,
                      engine="python").dropna(axis=1, how="all")[0].astype(int).to_numpy()
    return train, test, rul


# --------------------------------------------------------------------------- #
# Cleaning / labelling
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows, sort, and make sure there are no missing values."""
    df = df.drop_duplicates(subset=["unit", "cycle"]).sort_values(["unit", "cycle"])
    df = df.reset_index(drop=True)
    if df.isna().any().any():
        df = df.ffill().bfill()
    return df


def add_rul(train: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """RUL = (last cycle of engine) - (current cycle), capped at `cap`."""
    train = train.copy()
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train["rul_raw"] = max_cycle - train["cycle"]
    train["rul"] = train["rul_raw"].clip(upper=cap)
    return train


def detect_constant_columns(df: pd.DataFrame, cols: list[str], tol: float = 1e-8) -> list[str]:
    """Columns with a single unique value (or std <= tol to absorb float round-off, e.g. s5=14.62)."""
    return [c for c in cols if df[c].nunique() <= 1 or df[c].std() <= tol]


def select_features(train: pd.DataFrame, cfg: Config) -> tuple[list[str], list[str]]:
    """Return (feature_cols, dropped_constant_cols)."""
    candidates = (OP_COLS if cfg.use_op_settings else []) + SENSOR_COLS
    const = detect_constant_columns(train, OP_COLS + SENSOR_COLS)
    features = [c for c in candidates if c not in const]
    return features, const


# --------------------------------------------------------------------------- #
# Split, scale, window
# --------------------------------------------------------------------------- #
def split_units(units: np.ndarray, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    units = np.array(sorted(units))
    rng.shuffle(units)
    n_val = max(1, int(round(len(units) * val_frac)))
    return np.sort(units[n_val:]), np.sort(units[:n_val])


def fit_scaler(train_df: pd.DataFrame, features: list[str]) -> MinMaxScaler:
    """Fit on TRAINING-split engines only (no leakage from val/test)."""
    return MinMaxScaler(feature_range=(0, 1)).fit(train_df[features].to_numpy())


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[features] = scaler.transform(df[features].to_numpy())
    return out


def make_windows(df: pd.DataFrame, features: list[str], window: int,
                 target: str = "rul") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stride-1 sliding windows per engine. Returns X (N,window,F), y (N,), unit_ids (N,)."""
    X, y, u = [], [], []
    for unit, g in df.groupby("unit", sort=True):
        arr = g[features].to_numpy(dtype=np.float32)
        tgt = g[target].to_numpy(dtype=np.float32)
        for end in range(window, len(g) + 1):
            X.append(arr[end - window:end])
            y.append(tgt[end - 1])
            u.append(unit)
    return np.stack(X), np.asarray(y, dtype=np.float32), np.asarray(u)


def last_window(arr: np.ndarray, window: int) -> np.ndarray:
    """Last `window` rows of an (T,F) array; left-pad with the first row if T<window."""
    if len(arr) >= window:
        return arr[-window:]
    pad = np.repeat(arr[:1], window - len(arr), axis=0)
    return np.vstack([pad, arr])


def make_test_windows(test_scaled: pd.DataFrame, features: list[str], window: int):
    """One window (the last `window` cycles) per test engine."""
    X, units = [], []
    for unit, g in test_scaled.groupby("unit", sort=True):
        X.append(last_window(g[features].to_numpy(dtype=np.float32), window))
        units.append(unit)
    return np.stack(X).astype(np.float32), np.asarray(units)


def prepare(cfg: Config, data_dir: Path | str | None = None) -> dict:
    """Full preprocessing pipeline used by training, notebooks and tests."""
    train, test, rul_true = load_raw(cfg.subset, data_dir)
    train, test = clean(train), clean(test)
    train = add_rul(train, cfg.rul_cap)
    features, const = select_features(train, cfg)

    tr_units, va_units = split_units(train["unit"].unique(), cfg.val_frac, cfg.seed)
    tr_df = train[train["unit"].isin(tr_units)]
    va_df = train[train["unit"].isin(va_units)]

    scaler = fit_scaler(tr_df, features)
    tr_s, va_s = apply_scaler(tr_df, scaler, features), apply_scaler(va_df, scaler, features)
    te_s = apply_scaler(test, scaler, features)

    X_tr, y_tr, _ = make_windows(tr_s, features, cfg.window)
    X_va, y_va, _ = make_windows(va_s, features, cfg.window)
    X_te, te_units = make_test_windows(te_s, features, cfg.window)

    return dict(train=train, test=test, rul_true=rul_true, features=features,
                constant_cols=const, train_units=tr_units, val_units=va_units,
                scaler=scaler, X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va,
                X_test=X_te, test_units=te_units, test_scaled=te_s)
