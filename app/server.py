"""Local web server for the 3D dashboard.

Run:  python -m app.server     then open http://127.0.0.1:5000

Uses, in order of preference:
  data   -> data/processed/beats.npz, else synthetic demo beats
  model  -> models/ecgnet.pt (CNN), else models/baseline_random_forest.joblib, else none
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, send_from_directory

from src import config
from src.split import patient_split

STATIC = Path(__file__).parent / "static"
app = Flask(__name__, static_folder=None)


@lru_cache(maxsize=1)
def dataset():
    if config.BEATS_FILE.exists():
        d = np.load(config.BEATS_FILE)
        return d["X"], d["rr"], d["y"], d["patient"], "real"
    from src.synthetic import synthetic_dataset
    return (*synthetic_dataset(patients=config.DS2, beats_per_patient=500), "demo")


@lru_cache(maxsize=1)
def predictor():
    """Returns (name, fn) where fn(X, rr) -> class probabilities, or (None, None)."""
    ckpt = config.MODELS_DIR / "ecgnet.pt"
    if ckpt.exists():
        import torch
        from src.model import ECGNet

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = ECGNet(n_classes=len(config.CLASSES))
        model.load_state_dict(state["state_dict"])
        model.eval()

        def cnn(X, rr):
            rr_n = ((rr - state["rr_mean"]) / state["rr_std"]).astype(np.float32)
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 1024):
                    logits = model(torch.from_numpy(X[i:i + 1024]),
                                   torch.from_numpy(rr_n[i:i + 1024]))
                    out.append(torch.softmax(logits, 1).numpy())
            return np.concatenate(out)
        return "1D CNN", cnn

    rf = config.MODELS_DIR / "baseline_random_forest.joblib"
    if rf.exists():
        import joblib
        from src.features import build_features

        model = joblib.load(rf)

        def forest(X, rr):
            proba = model.predict_proba(build_features(X, rr))
            full = np.zeros((len(X), len(config.CLASSES)), np.float32)
            full[:, model.classes_] = proba
            return full
        return "Random forest", forest
    return None, None


def test_patients():
    *_, patient, _ = dataset()
    return [p for p in config.DS2 if (patient == p).any()]


@lru_cache(maxsize=64)
def patient_payload(pid):
    X, rr, y, patient, _ = dataset()
    sel = patient == pid
    if not sel.any():
        return None
    Xp, rrp, yp = X[sel], rr[sel], y[sel]
    name, fn = predictor()
    payload = {
        "id": pid,
        "beats": np.round(Xp[:, ::2].astype(float), 3).tolist(),        # 125 points per beat
        "labels": yp.tolist(),
        "rr": np.round(rrp[:, 0].astype(float), 3).tolist(),            # seconds since previous beat
    }
    if fn is not None:
        proba = fn(Xp, rrp)
        payload["pred"] = proba.argmax(1).tolist()
        payload["proba"] = np.round(proba.astype(float), 3).tolist()
    return json.dumps(payload)


@app.get("/api/status")
def status():
    *_, mode = dataset()
    name, _ = predictor()
    return jsonify(mode=mode, model=name, classes=config.CLASSES,
                   fs=config.FS, window_before=config.WINDOW_BEFORE)


@app.get("/api/patients")
def patients():
    _, _, y, patient, _ = dataset()
    out = []
    for p in test_patients():
        counts = np.bincount(y[patient == p], minlength=len(config.CLASSES))
        out.append({"id": p, "n": int(counts.sum()), "counts": counts.tolist()})
    return jsonify(out)


@app.get("/api/patients/<pid>")
def patient(pid):
    if pid not in test_patients():
        abort(404)
    return app.response_class(patient_payload(pid), mimetype="application/json")


@app.get("/api/metrics")
def metrics():
    out = {}
    for name in ["cnn", "baseline_random_forest", "baseline_logreg"]:
        f = config.REPORTS_DIR / f"{name}_metrics.json"
        if f.exists():
            out[name] = json.loads(f.read_text())
    f = config.REPORTS_DIR / "leakage_demo.json"
    if f.exists():
        out["leakage"] = json.loads(f.read_text())
    return jsonify(out)


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:path>")
def static_files(path):
    return send_from_directory(STATIC, path)


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False)
