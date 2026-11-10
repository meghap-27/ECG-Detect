"""Show how a beat-level random split inflates results compared to a patient-level split.

Same features, same model; only the split changes. With a random split, beats from
the same patient land in both train and test, so the model can recognise the
*patient's* waveform instead of the *arrhythmia*.
Run:  python -m src.leakage_demo
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src import config
from src.data import load_dataset
from src.evaluate import compute_metrics
from src.features import build_features
from src.split import patient_split, random_beat_split


def _fit_eval(feats, y, train, test):
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                   class_weight="balanced_subsample", n_jobs=-1,
                                   random_state=config.SEED)
    model.fit(feats[train], y[train])
    return compute_metrics(y[test], model.predict(feats[test]))


def run(X, rr, y, patient, reports_dir=config.REPORTS_DIR):
    feats = build_features(X, rr)
    pm = patient_split(patient)
    rm = random_beat_split(y)
    results = {
        "random_beat_split": _fit_eval(feats, y, rm["train"], rm["test"]),
        "patient_split": _fit_eval(feats, y, pm["train"] | pm["val"], pm["test"]),
    }

    print(f"{'metric':<22}{'random split':>14}{'patient split':>15}")
    rows = [("accuracy", lambda m: m["accuracy"]),
            ("macro-F1", lambda m: m["macro_f1"])]
    rows += [(f"{c} sensitivity", lambda m, c=c: m["per_class"][c]["sensitivity"])
             for c in ["S", "V", "F"]]
    for label, get in rows:
        print(f"{label:<22}{get(results['random_beat_split']):>14.3f}"
              f"{get(results['patient_split']):>15.3f}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "leakage_demo.json").write_text(json.dumps(results, indent=2))

    names = [r[0] for r in rows]
    a = [get(results["random_beat_split"]) for _, get in rows]
    b = [get(results["patient_split"]) for _, get in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, a, 0.4, label="Random beat split (leaky)", color="#c44e52")
    ax.bar(x + 0.2, b, 0.4, label="Patient-level split (honest)", color="#4c72b0")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Same model, different split: how leakage inflates results")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(reports_dir / "leakage_demo.png", dpi=150)
    plt.close(fig)
    return results


if __name__ == "__main__":
    run(*load_dataset())
