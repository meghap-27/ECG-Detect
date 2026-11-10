"""Hand-crafted features for the classical baseline models."""
import numpy as np
from scipy.stats import kurtosis, skew

from src import config


def morphology_features(X):
    """Simple shape descriptors of each z-scored beat."""
    r = config.WINDOW_BEFORE
    qrs = X[:, r - 18:r + 18]            # ~100 ms around the R-peak
    feats = np.column_stack([
        X.max(axis=1), X.min(axis=1),
        X.argmax(axis=1), X.argmin(axis=1),
        skew(X, axis=1), kurtosis(X, axis=1),
        (qrs ** 2).sum(axis=1) / ((X ** 2).sum(axis=1) + 1e-8),   # QRS energy share
        np.abs(np.diff(qrs, axis=1)).max(axis=1),                 # steepest QRS slope
        X[:, r + 40:].mean(axis=1),                               # T-wave region level
    ])
    downsampled = X[:, ::10]              # coarse waveform, 25 points
    return np.hstack([feats, downsampled])


def build_features(X, rr):
    return np.hstack([rr, morphology_features(X)]).astype(np.float32)
