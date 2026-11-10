"""Train/validation/test splits.

`patient_split` is the honest one: every patient's beats live in exactly one split.
`random_beat_split` is the leaky one most tutorials use; it exists only so
`src.leakage_demo` can show how much it inflates the results.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from src import config


def patient_split(patient, val_patients=config.VAL_PATIENTS,
                  train_pool=config.DS1, test_pool=config.DS2):
    """Boolean masks for train (DS1 minus val), val (subset of DS1) and test (DS2)."""
    patient = np.asarray(patient)
    train_ids = [p for p in train_pool if p not in val_patients]
    masks = {
        "train": np.isin(patient, train_ids),
        "val": np.isin(patient, val_patients),
        "test": np.isin(patient, test_pool),
    }
    assert_no_patient_overlap(patient, masks)
    return masks


def assert_no_patient_overlap(patient, masks):
    seen = {}
    for name, m in masks.items():
        for p in np.unique(patient[m]):
            if p in seen:
                raise ValueError(f"Patient {p} is in both '{seen[p]}' and '{name}'")
            seen[p] = name


def random_beat_split(y, test_size=0.3, seed=config.SEED):
    """Stratified split by beat, ignoring patients (leaks patient identity)."""
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=y,
                                           random_state=seed)
    train, test = np.zeros(len(y), bool), np.zeros(len(y), bool)
    train[train_idx], test[test_idx] = True, True
    return {"train": train, "test": test}
