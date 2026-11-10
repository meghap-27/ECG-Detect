"""Classical baselines (logistic regression, random forest) on hand-crafted features.

Always build a simple baseline first: it tells you how much the CNN actually adds.
Run:  python -m src.baseline
"""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data import load_dataset
from src.evaluate import save_report
from src.features import build_features
from src.split import patient_split


def make_models(seed=config.SEED):
    return {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=seed),
    }


def run(X, rr, y, patient, reports_dir=config.REPORTS_DIR, models_dir=config.MODELS_DIR):
    masks = patient_split(patient)
    feats = build_features(X, rr)
    train = masks["train"] | masks["val"]   # baselines need no early stopping
    results = {}
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, model in make_models().items():
        model.fit(feats[train], y[train])
        pred = model.predict(feats[masks["test"]])
        results[name] = save_report(f"baseline_{name}", y[masks["test"]], pred, reports_dir)
        joblib.dump(model, models_dir / f"baseline_{name}.joblib")
    return results


if __name__ == "__main__":
    run(*load_dataset())
