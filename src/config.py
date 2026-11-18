"""Central configuration: paths, column names and hyper-parameters."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
FIG_DIR = ARTIFACT_DIR / "figures"

SUBSET = "FD001"

INDEX_COLS = ["unit", "cycle"]
OP_COLS = [f"op{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + OP_COLS + SENSOR_COLS  # 26 columns

# Human-readable sensor names (from Saxena et al., PHM'08, Table 2)
SENSOR_NAMES = {
    "s1": "T2 - Fan inlet temp", "s2": "T24 - LPC outlet temp",
    "s3": "T30 - HPC outlet temp", "s4": "T50 - LPT outlet temp",
    "s5": "P2 - Fan inlet pressure", "s6": "P15 - Bypass-duct pressure",
    "s7": "P30 - HPC outlet pressure", "s8": "Nf - Physical fan speed",
    "s9": "Nc - Physical core speed", "s10": "epr - Engine pressure ratio",
    "s11": "Ps30 - HPC outlet static pressure", "s12": "phi - Fuel flow / Ps30",
    "s13": "NRf - Corrected fan speed", "s14": "NRc - Corrected core speed",
    "s15": "BPR - Bypass ratio", "s16": "farB - Burner fuel-air ratio",
    "s17": "htBleed - Bleed enthalpy", "s18": "Nf_dmd - Demanded fan speed",
    "s19": "PCNfR_dmd - Demanded corrected fan speed",
    "s20": "W31 - HPT coolant bleed", "s21": "W32 - LPT coolant bleed",
}


@dataclass
class Config:
    subset: str = SUBSET
    rul_cap: int = 125          # piece-wise linear RUL cap
    window: int = 30            # sliding-window length (cycles)
    val_frac: float = 0.20      # fraction of ENGINES held out for validation
    use_op_settings: bool = False  # FD001 has ONE operating condition -> op settings are pure noise
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 100
    patience: int = 12          # early-stopping patience (epochs)
    seed: int = 42
    device: str = "auto"        # auto | cpu | cuda | mps

    def to_dict(self) -> dict:
        return asdict(self)
