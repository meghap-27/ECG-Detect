"""Train the 1D CNN with a patient-level split, early stopping on validation macro-F1.

Run:  python -m src.train            (full training)
      python -m src.train --epochs 3 (quick check)
"""
import argparse
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.data import load_dataset
from src.evaluate import compute_metrics, save_report
from src.model import ECGNet
from src.split import patient_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y, n_classes):
    """Inverse square-root frequency: boosts rare classes without letting them dominate."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    w[counts == 0] = 0.0
    return torch.tensor(w / w[counts > 0].mean(), dtype=torch.float32)


def augment(beat):
    """Light augmentation: random amplitude scaling, small time shift, Gaussian noise."""
    scale = torch.empty(beat.size(0), 1, device=beat.device).uniform_(0.9, 1.1)
    shift = random.randint(-5, 5)
    beat = torch.roll(beat * scale, shifts=shift, dims=1)
    return beat + 0.03 * torch.randn_like(beat)


@torch.no_grad()
def predict(model, X, rr, device, batch_size=1024):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        rb = torch.from_numpy(rr[i:i + batch_size]).to(device)
        preds.append(model(xb, rb).argmax(1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)


def run(X, rr, y, patient, epochs=30, batch_size=256, lr=1e-3, patience=6,
        seed=config.SEED, reports_dir=config.REPORTS_DIR, models_dir=config.MODELS_DIR):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    n_classes = len(config.CLASSES)
    m = patient_split(patient)

    # Standardise RR features with training statistics only (no peeking at test).
    rr_mean = rr[m["train"]].mean(axis=0)
    rr_std = rr[m["train"]].std(axis=0) + 1e-8
    rr_n = ((rr - rr_mean) / rr_std).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X[m["train"]]), torch.from_numpy(rr_n[m["train"]]),
                             torch.from_numpy(y[m["train"]]))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = ECGNet(n_classes=n_classes, n_rr=rr.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y[m["train"]], n_classes).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=2)

    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt = models_dir / "ecgnet.pt"
    best_f1, bad_epochs, history = -1.0, 0, []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, rb, yb in loader:
            xb, rb, yb = xb.to(device), rb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(augment(xb), rb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)

        val_pred = predict(model, X[m["val"]], rr_n[m["val"]], device)
        val_f1 = compute_metrics(y[m["val"]], val_pred)["macro_f1"]
        scheduler.step(val_f1)
        train_loss = total / len(train_ds)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_f1})
        print(f"epoch {epoch:>3} | train loss {train_loss:.4f} | val macro-F1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            torch.save({"state_dict": model.state_dict(), "rr_mean": rr_mean,
                        "rr_std": rr_std, "classes": config.CLASSES}, ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping (best val macro-F1 {best_f1:.3f})")
                break

    # Final, one-time evaluation on unseen test patients with the best checkpoint.
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    test_pred = predict(model, X[m["test"]], rr_n[m["test"]], device)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "cnn_history.json").write_text(json.dumps(history, indent=2))
    return save_report("cnn", y[m["test"]], test_pred, reports_dir)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(*load_dataset(), epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        patience=a.patience, seed=a.seed)
