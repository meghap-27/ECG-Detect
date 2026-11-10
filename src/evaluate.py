"""Metrics, reports and plots shared by every model.

Accuracy is misleading here (~89% of test beats are N), so we report per-class
sensitivity (recall) and positive predictive value (precision), as the AAMI
standard does, plus macro-F1 as the single headline number.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from src import config

LABELS = list(range(len(config.CLASSES)))


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    report = classification_report(y_true, y_pred, labels=LABELS,
                                   target_names=config.CLASSES,
                                   zero_division=0, output_dict=True)
    present = [c for c in LABELS if (np.asarray(y_true) == c).any()]
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        # macro-F1 over classes that actually occur in this split
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro",
                                   zero_division=0)),
        "per_class": {
            c: {"sensitivity": report[c]["recall"], "ppv": report[c]["precision"],
                "f1": report[c]["f1-score"], "support": int(report[c]["support"])}
            for c in config.CLASSES
        },
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion(cm, title, path):
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(LABELS, config.CLASSES)
    ax.set_yticks(LABELS, config.CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in LABELS:
        for j in LABELS:
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=8, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_summary(name, m):
    lines = [f"== {name} ==",
             f"accuracy {m['accuracy']:.3f} | balanced acc {m['balanced_accuracy']:.3f} "
             f"| macro-F1 {m['macro_f1']:.3f}",
             f"{'class':<6}{'Se':>8}{'PPV':>8}{'F1':>8}{'n':>8}"]
    for c, v in m["per_class"].items():
        lines.append(f"{c:<6}{v['sensitivity']:>8.3f}{v['ppv']:>8.3f}{v['f1']:>8.3f}"
                     f"{v['support']:>8}")
    return "\n".join(lines)


def save_report(name, y_true, y_pred, reports_dir=config.REPORTS_DIR):
    """Write <name>_metrics.json, <name>_summary.txt and <name>_confusion.png."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    m = compute_metrics(y_true, y_pred)
    summary = format_summary(name, m)
    (reports_dir / f"{name}_metrics.json").write_text(json.dumps(m, indent=2))
    (reports_dir / f"{name}_summary.txt").write_text(summary + "\n")
    plot_confusion(m["confusion_matrix"], f"{name} (test patients)",
                   reports_dir / f"{name}_confusion.png")
    print(summary)
    return m
