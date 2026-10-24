"""Dataset loaders.

aircraft : NASA C-MAPSS FD001 (turbofan engines, run to failure)
factory  : NASA/IMS bearing run-to-failure data (vibration files, 10 minutes apart)

Both loaders return the same `Bundle`, so train.py does not care which machine it is training.
Life left ("rul") is capped at `cap`, and sensor/feature columns are smoothed with a rolling mean.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config


@dataclass
class Bundle:
    domain: str
    features: list
    feature_labels: dict
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    test_truth: np.ndarray = None          # aircraft only: true life left at the last cycle of each test engine
    heldout: dict = field(default_factory=dict)   # name -> (label, dataframe) run-to-failure units never used to fit the ANN
    long_labels: dict = field(default_factory=dict)
    note: str = ""


def _smooth(df, features, window):
    df = df.copy()
    df[features] = df.groupby("unit")[features].transform(lambda s: s.rolling(window, min_periods=1).mean())
    return df


# --------------------------------------------------------------------------- C-MAPSS
CMAPSS_COLS = ["unit", "t", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def _read_cmapss(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLS)


def load_aircraft(data_dir, subset="FD001", seed=0, val_fraction=0.2):
    cfg = config.DOMAINS["aircraft"]
    data_dir = Path(data_dir)
    for name in (f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"):
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"{data_dir / name} not found. See README.md, section 'Datasets'.")
    train = _read_cmapss(data_dir / f"train_{subset}.txt")
    test = _read_cmapss(data_dir / f"test_{subset}.txt")
    test_truth = pd.read_csv(data_dir / f"RUL_{subset}.txt", header=None)[0].to_numpy(dtype=float)

    features = [f"s{i}" for i in range(1, 22)]
    train["rul"] = (train.groupby("unit")["t"].transform("max") - train["t"]).clip(upper=cfg["cap"])
    test["rul"] = np.nan
    train = _smooth(train, features, cfg["smooth_window"])
    test = _smooth(test, features, cfg["smooth_window"])

    units = np.array(sorted(train["unit"].unique()))
    rng = np.random.default_rng(seed)
    val_units = set(rng.choice(units, size=int(round(len(units) * val_fraction)), replace=False).tolist())
    val = train[train["unit"].isin(val_units)].reset_index(drop=True)
    fit = train[~train["unit"].isin(val_units)].reset_index(drop=True)
    heldout = {int(u): (f"Engine {int(u)}", g.reset_index(drop=True)) for u, g in val.groupby("unit")}
    labels = {f: f"S{i}" for i, f in enumerate(features, 1)}
    labels_long = {f: f"S{i} {config.CMAPSS_SENSORS[i]}" for i, f in enumerate(features, 1)}
    note = (f"C-MAPSS {subset}: {len(units) - len(val_units)} engines to fit, {len(val_units)} held out, "
            f"{test['unit'].nunique()} official test engines")
    return Bundle("aircraft", features, labels, fit, val, test, test_truth, heldout, labels_long, note)


# --------------------------------------------------------------------------- IMS bearings
# (run name, folder names to look for, channels (0-based) of the bearing that failed)
IMS_RUNS = [
    ("set1_bearing3", ["1st_test"], [4, 5]),
    ("set1_bearing4", ["1st_test"], [6, 7]),
    ("set2_bearing1", ["2nd_test"], [0]),
    ("set3_bearing3", ["3rd_test", "4th_test"], [2]),
]
IMS_FEATURES = ["rms", "peak", "crest", "kurtosis", "skew", "std", "ptp", "shape", "impulse", "e_low", "e_mid", "e_high"]
IMS_LABELS = {"rms": "RMS", "peak": "Peak", "crest": "Crest", "kurtosis": "Kurt", "skew": "Skew", "std": "Std",
              "ptp": "P2P", "shape": "Shape", "impulse": "Imp", "e_low": "E-lo", "e_mid": "E-mid", "e_high": "E-hi"}
IMS_LONG = {"rms": "Vibration RMS", "peak": "Vibration peak", "crest": "Crest factor", "kurtosis": "Kurtosis",
            "skew": "Skewness", "std": "Standard deviation", "ptp": "Peak-to-peak", "shape": "Shape factor",
            "impulse": "Impulse factor", "e_low": "Low-band energy", "e_mid": "Mid-band energy", "e_high": "High-band energy"}


def _vibration_features(x):
    """Features of one vibration snapshot (1-D array)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    peak = np.max(np.abs(x))
    mean_abs = np.mean(np.abs(x)) + 1e-12
    std = x.std() + 1e-12
    spec = np.abs(np.fft.rfft(x)) ** 2
    n = len(spec)
    lo, mid = n // 10, n // 2           # 0-10 %, 10-50 %, 50-100 % of the spectrum
    total = spec.sum() + 1e-12
    return {
        "rms": rms, "peak": peak, "crest": peak / rms,
        "kurtosis": np.mean((x / std) ** 4), "skew": np.mean((x / std) ** 3),
        "std": std, "ptp": x.max() - x.min(), "shape": rms / mean_abs, "impulse": peak / mean_abs,
        "e_low": spec[:lo].sum() / total, "e_mid": spec[lo:mid].sum() / total, "e_high": spec[mid:].sum() / total,
    }


def _extract_ims_features(data_dir, cache_csv):
    cache_csv = Path(cache_csv)
    if cache_csv.exists():
        return pd.read_csv(cache_csv)
    rows = []
    folders_done = {}
    for name, folder_names, channels in IMS_RUNS:
        folder = next((Path(data_dir) / f for f in folder_names if (Path(data_dir) / f).is_dir()), None)
        if folder is None:
            print(f"  [skip] {name}: folder {folder_names} not found in {data_dir}")
            continue
        # some downloads have an extra sub-folder (e.g. 4th_test/txt)
        files = sorted(p for p in folder.rglob("*") if p.is_file() and not p.name.startswith("."))
        print(f"  {name}: {len(files)} files in {folder}")
        for step, f in enumerate(files):
            arr = pd.read_csv(f, sep=r"\s+", header=None).to_numpy()
            feats = [_vibration_features(arr[:, c]) for c in channels]
            rows.append({"unit": name, "t": step + 1, **{k: float(np.mean([d[k] for d in feats])) for k in IMS_FEATURES}})
    if not rows:
        raise FileNotFoundError(f"No IMS bearing folders found in {data_dir}. See README.md, section 'Datasets'.")
    df = pd.DataFrame(rows)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return df


def load_factory(data_dir, test_run="set2_bearing1", val_run="set1_bearing4", healthy_ratio=3, **_):
    cfg = config.DOMAINS["factory"]
    df = _extract_ims_features(data_dir, Path(data_dir) / "ims_features.csv")
    df["rul"] = (df.groupby("unit")["t"].transform("max") - df["t"]).clip(upper=cfg["cap"])
    df = _smooth(df, IMS_FEATURES, cfg["smooth_window"])
    runs = list(df["unit"].unique())
    if len(runs) < 3:
        raise ValueError(f"Need at least 3 run-to-failure bearings (train, validation, test); found {runs}")
    if test_run not in runs:
        test_run = runs[-1]
    if val_run not in runs or val_run == test_run:
        val_run = next(r for r in reversed(runs) if r != test_run)
    fit_runs = [r for r in runs if r not in (test_run, val_run)]

    def thin(frame):
        """Long healthy stretches would swamp training: keep every k-th healthy row."""
        deg = frame[frame["rul"] < cfg["cap"]]
        healthy = frame[frame["rul"] >= cfg["cap"]]
        keep = max(1, int(np.ceil(len(healthy) / max(1, healthy_ratio * len(deg)))))
        return pd.concat([deg, healthy.iloc[::keep]]).sort_values(["unit", "t"]).reset_index(drop=True)

    fit = thin(df[df["unit"].isin(fit_runs)])
    val = df[df["unit"] == val_run].reset_index(drop=True)
    test = df[df["unit"] == test_run].reset_index(drop=True)
    heldout = {val_run: (val_run.replace("_", " "), val), test_run: (test_run.replace("_", " "), test)}
    note = f"IMS bearings: fit on {fit_runs}, validate on {val_run}, test on {test_run}"
    return Bundle("factory", IMS_FEATURES, IMS_LABELS, fit, val, test, None, heldout, IMS_LONG, note)


def load_domain(domain, data_dir, **kw):
    if domain == "aircraft":
        return load_aircraft(data_dir, **{k: v for k, v in kw.items() if k in ("subset", "seed")})
    if domain == "factory":
        return load_factory(data_dir, **{k: v for k, v in kw.items() if k in ("test_run", "val_run")})
    raise ValueError(domain)
