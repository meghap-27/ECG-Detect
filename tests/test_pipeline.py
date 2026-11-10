"""Fast smoke tests on synthetic beats (no download needed).

Run:  python -m pytest -q
"""
import numpy as np
import pytest

from src import config
from src.data import extract_beats, rr_features
from src.split import patient_split, random_beat_split


def synthetic_dataset(beats_per_patient=60, seed=0):
    """Fake beats: a Gaussian 'QRS' whose width and timing depend on the class."""
    rng = np.random.default_rng(seed)
    t = np.arange(config.BEAT_LEN)
    X, rr, y, patient = [], [], [], []
    for p in config.DS1 + config.DS2:
        for _ in range(beats_per_patient):
            c = rng.choice(5, p=[0.6, 0.15, 0.15, 0.05, 0.05])
            width = [6, 6, 18, 12, 9][c]
            beat = np.exp(-((t - config.WINDOW_BEFORE) ** 2) / (2 * width ** 2))
            beat += 0.1 * rng.standard_normal(config.BEAT_LEN)
            X.append((beat - beat.mean()) / beat.std())
            pre = 0.8 * (0.7 if c in (1, 2) else 1.0)
            rr.append([pre, 0.8, pre / 0.8, 0.8 / pre])
            y.append(c)
            patient.append(p)
    return (np.array(X, np.float32), np.array(rr, np.float32), np.array(y, np.int64),
            np.array(patient))


def test_patient_split_has_no_overlap():
    _, _, _, patient = synthetic_dataset(5)
    m = patient_split(patient)
    sets = {k: set(patient[v]) for k, v in m.items()}
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])
    assert not (sets["train"] & sets["val"])


def test_random_split_does_leak():
    _, _, y, patient = synthetic_dataset(20)
    m = random_beat_split(y)
    assert set(patient[m["train"]]) & set(patient[m["test"]])


def test_rr_features_shape_and_values():
    peaks = np.array([0, 360, 720, 1000])
    f = rr_features(peaks)
    assert f.shape == (4, 4)
    assert np.isnan(f[0, 0]) and np.isnan(f[-1, 1])
    assert f[1, 0] == pytest.approx(1.0)


def test_extract_beats_skips_non_beat_annotations():
    sig = np.sin(np.linspace(0, 60, 5000))
    peaks = [200, 500, 600, 900, 1200, 1500]
    symbols = ["N", "+", "V", "N", "A", "N"]
    X, rr, y, p = extract_beats(sig, peaks, symbols, "999")
    assert X.shape[1] == config.BEAT_LEN
    assert len(X) == len(y) == len(rr) == len(p)
    assert set(y) <= {0, 1, 2}


def test_baseline_and_cnn_run(tmp_path):
    from src import baseline, train

    data = synthetic_dataset(30)
    res = baseline.run(*data, reports_dir=tmp_path, models_dir=tmp_path)
    assert res["random_forest"]["macro_f1"] > 0.5
    m = train.run(*data, epochs=2, reports_dir=tmp_path, models_dir=tmp_path)
    assert (tmp_path / "cnn_confusion.png").exists()
    assert 0.0 <= m["macro_f1"] <= 1.0
