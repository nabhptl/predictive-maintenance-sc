"""Shared settings for both machine types (aircraft engine, factory machine)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

# Official meaning of the 21 C-MAPSS sensor columns (Saxena et al., 2008).
CMAPSS_SENSORS = {
    1: "Fan inlet temperature", 2: "LPC outlet temperature", 3: "HPC outlet temperature",
    4: "LPT outlet temperature", 5: "Fan inlet pressure", 6: "Bypass-duct pressure",
    7: "HPC outlet pressure", 8: "Physical fan speed", 9: "Physical core speed",
    10: "Engine pressure ratio", 11: "HPC outlet static pressure", 12: "Fuel flow / Ps30",
    13: "Corrected fan speed", 14: "Corrected core speed", 15: "Bypass ratio",
    16: "Burner fuel-air ratio", 17: "Bleed enthalpy", 18: "Demanded fan speed",
    19: "Demanded corrected fan speed", 20: "HPT coolant bleed", 21: "LPT coolant bleed",
}

# Everything that differs between machine types lives here. The code does not change.
DOMAINS = {
    "aircraft": {
        "label": "Aircraft engine",
        "unit": "flights",
        "unit_singular": "flight",
        "cap": 125,               # life-left label is capped here (C-MAPSS convention)
        "smooth_window": 5,       # rolling mean over sensor readings
        "feature_penalty": 0.15,  # GA fitness: RMSE + penalty * number_of_features
        "replay_max_len": None,   # dashboard replays the full run
    },
    "factory": {
        "label": "Factory machine",
        "unit": "readings",       # one reading = one 10-minute vibration snapshot (IMS bearings)
        "unit_singular": "reading",
        "cap": 150,
        "smooth_window": 5,
        "feature_penalty": 0.15,
        "replay_max_len": 400,    # dashboard replays the last 400 readings of the run
    },
}

# Fuzzy alert thresholds on the score in [0, 1]
WARNING_AT = 0.30
CRITICAL_AT = 0.70
