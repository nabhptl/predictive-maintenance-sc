"""Fuzzy risk engine and wear-rate estimator (pure numpy, no extra libraries).

Inputs
  life_left : predicted remaining life from the neural network, rescaled to 0-125
  wear      : how fast the sensors are drifting (0 = steady, 1 = fastest seen in training)
Output
  score     : risk score in [0, 1]  ->  Normal / Warning / Critical

The membership functions and rules are identical to the ones in the dashboard JavaScript,
so the numbers on screen match what Python computes.
"""
import numpy as np

import config

LEVELS = ["Normal", "Warning", "Critical"]

# (rule text, weight of the output set: Normal=0, Warning=0.5, Critical=1)
RULES = [
    ("IF life left is long AND wear is slow THEN Normal", 0.0),
    ("IF life left is long AND wear is fast THEN Warning", 0.5),
    ("IF life left is medium AND wear is slow THEN Warning", 0.5),
    ("IF life left is medium AND wear is fast THEN Critical", 1.0),
    ("IF life left is short THEN Critical", 1.0),
]
WEIGHTS = np.array([w for _, w in RULES])

# trapezoid corners (a, b, c, d)
LIFE_SETS = {"short": (-1, 0, 15, 35), "medium": (20, 40, 60, 90), "long": (60, 100, 200, 300)}
WEAR_SETS = {"slow": (-1, 0, 0.25, 0.6), "fast": (0.25, 0.6, 1, 2)}


def trap(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    return np.clip(np.minimum(np.minimum((x - a) / (b - a), 1.0), (d - x) / (d - c)), 0.0, 1.0)


def firing(life_left, wear):
    """Strength (0-1) of each of the 5 rules, shape (n, 5)."""
    sh, md, lg = (trap(life_left, *LIFE_SETS[k]) for k in ("short", "medium", "long"))
    sl, fa = (trap(wear, *WEAR_SETS[k]) for k in ("slow", "fast"))
    return np.stack([np.minimum(lg, sl), np.minimum(lg, fa), np.minimum(md, sl), np.minimum(md, fa), sh], axis=-1)


def risk_score(life_left, wear):
    f = firing(life_left, wear)
    den = f.sum(axis=-1)
    num = (f * WEIGHTS).sum(axis=-1)
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)


def level_of(score):
    score = np.asarray(score)
    return np.where(score < config.WARNING_AT, 0, np.where(score < config.CRITICAL_AT, 1, 2))


def ema(x, alpha):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    if len(x) == 0:
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def alert_series(pred, wear, cap, alpha=0.35):
    """From raw ANN predictions of one unit (time order) to smoothed prediction, score and level."""
    ps = ema(pred, alpha)
    score = risk_score(ps / cap * 125.0, wear)
    return ps, score, level_of(score)


def threshold_levels(pred, cap, alpha=0.35):
    """Baseline without fuzzy logic: fixed cut-offs on the smoothed predicted life left."""
    norm = ema(pred, alpha) / cap * 125.0
    return np.where(norm < 30, 2, np.where(norm < 70, 1, 0))


class WearEstimator:
    """Wear rate = how quickly a health index built from the selected sensors is rising.

    The health index is the mean of the selected (min-max scaled) sensors, each flipped so
    that it increases as the unit wears out. The 10-step drift is divided by the 95th
    percentile of drifts seen in training, so 1.0 means 'as fast as the fastest training wear'.
    """

    def __init__(self, window=10, smooth=0.3):
        self.window, self.smooth = window, smooth
        self.signs, self.ref = None, 1.0

    def _hi(self, X):
        return X @ self.signs / X.shape[1]

    def fit(self, X, rul, units):
        corr = np.array([np.corrcoef(X[:, j], rul)[0, 1] if X[:, j].std() > 1e-9 else 0.0 for j in range(X.shape[1])])
        self.signs = -np.sign(np.nan_to_num(corr))
        self.signs[self.signs == 0] = 1.0
        drifts = []
        hi = self._hi(X)
        w = self.window
        for u in np.unique(units):
            h = hi[units == u]
            if len(h) > w:
                drifts.append(np.clip((h[w:] - h[:-w]) / w, 0, None))
        self.ref = float(np.percentile(np.concatenate(drifts), 95)) + 1e-9
        return self

    def transform(self, X):
        """X: one unit, rows in time order. Returns wear in [0, 1] per row."""
        hi = self._hi(X)
        idx = np.arange(len(hi))
        lag = np.minimum(self.window, idx)
        drift = np.where(lag > 0, (hi - hi[idx - lag]) / np.maximum(lag, 1), 0.0)
        return ema(np.clip(drift / self.ref, 0, 1), self.smooth)
