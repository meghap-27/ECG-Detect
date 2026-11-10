"""Realistic-looking synthetic heartbeats for demo mode and tests.

Each beat is built from Gaussian P, QRS and T waves (a simplified version of the
McSharry ECG model). It lets the dashboard run before the real data is downloaded.
It is NOT a substitute for MIT-BIH: never report results on it.
"""
import numpy as np

from src import config

# (P amp, QRS width, T amp, RR prematurity factor) per class
_SHAPES = {
    0: dict(p=0.15, qrs=0.010, r=1.0, t=0.30, rr=1.00),    # N
    1: dict(p=-0.08, qrs=0.010, r=0.95, t=0.28, rr=0.72),  # S: abnormal P, premature
    2: dict(p=0.0, qrs=0.032, r=1.4, t=-0.45, rr=0.75),    # V: no P, wide QRS, inverted T
    3: dict(p=0.08, qrs=0.020, r=1.15, t=-0.10, rr=0.90),  # F: halfway between N and V
    4: dict(p=0.05, qrs=0.016, r=0.70, t=0.15, rr=1.00),   # Q
}


def synth_beat(cls, rng, patient_style):
    t = (np.arange(config.BEAT_LEN) - config.WINDOW_BEFORE) / config.FS   # seconds
    s = _SHAPES[cls]
    amp, tilt = patient_style

    def g(a, mu, sd):
        return a * np.exp(-((t - mu) ** 2) / (2 * sd ** 2))

    beat = (g(s["p"], -0.18, 0.025)
            + g(-0.12 * s["r"], -0.025, s["qrs"] * 0.8)
            + g(s["r"] * amp, 0.0, s["qrs"])
            + g(-0.2 * s["r"], 0.03, s["qrs"] * 0.9)
            + g(s["t"], 0.25, 0.045))
    beat += tilt * t + 0.03 * rng.standard_normal(len(t))
    return beat


def synthetic_dataset(patients=None, beats_per_patient=400, seed=0):
    rng = np.random.default_rng(seed)
    patients = patients or (config.DS1 + config.DS2)
    X, rr, y, pat = [], [], [], []
    for p in patients:
        style = (rng.uniform(0.8, 1.2), rng.uniform(-0.3, 0.3))
        base_rr = rng.uniform(0.65, 1.0)
        mix = rng.dirichlet([30, 2, 3, 0.5, 0.3])
        prev = base_rr
        for _ in range(beats_per_patient):
            c = int(rng.choice(5, p=mix))
            pre = base_rr * _SHAPES[c]["rr"] * rng.uniform(0.95, 1.05)
            post = base_rr * (1.25 if c in (1, 2) else 1.0) * rng.uniform(0.95, 1.05)
            beat = synth_beat(c, rng, style)
            X.append((beat - beat.mean()) / (beat.std() + 1e-8))
            rr.append([pre, post, pre / base_rr, post / pre])
            y.append(c)
            pat.append(p)
            prev = pre
    return (np.array(X, np.float32), np.array(rr, np.float32), np.array(y, np.int64),
            np.array(pat))
