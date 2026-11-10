"""Download MIT-BIH, filter the ECG, segment heartbeats and save them as one .npz file.

Run:  python -m src.data
"""
import numpy as np
from scipy.signal import butter, filtfilt

from src import config


def download_mitdb(dest=config.DATA_RAW):
    """Download the MIT-BIH Arrhythmia Database from PhysioNet (~100 MB), once."""
    import wfdb

    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "100.hea").exists():
        print(f"MIT-BIH already present in {dest}")
        return
    print("Downloading MIT-BIH Arrhythmia Database from PhysioNet ...")
    wfdb.dl_database("mitdb", str(dest))


def bandpass(signal, fs=config.FS, low=0.5, high=40.0, order=3):
    """Remove baseline wander (<0.5 Hz) and high-frequency/powerline noise (>40 Hz)."""
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, signal)


def rr_features(r_peaks, fs=config.FS, local_n=10):
    """Per-beat RR-interval features, normalised per patient so they transfer across people.

    Returns an (n_beats, 4) array: [pre_RR, post_RR, pre_RR / local_avg, post_RR / pre_RR].
    The first and last beats get NaN (no neighbour) and are dropped later.
    """
    rr = np.diff(r_peaks) / fs
    n = len(r_peaks)
    pre = np.full(n, np.nan)
    post = np.full(n, np.nan)
    pre[1:] = rr
    post[:-1] = rr

    local = np.full(n, np.nan)
    for i in range(1, n):
        lo = max(1, i - local_n + 1)
        local[i] = np.mean(pre[lo:i + 1])

    with np.errstate(divide="ignore", invalid="ignore"):
        feats = np.column_stack([pre, post, pre / local, post / pre])
    return feats


def extract_beats(signal, r_peaks, symbols, patient, fs=config.FS):
    """Cut a fixed window around each annotated beat and z-score it.

    Only annotations that are real heartbeats (keys of AAMI_MAP) are used; rhythm
    and noise markers are ignored.
    """
    keep = np.array([s in config.AAMI_MAP for s in symbols])
    r_peaks = np.asarray(r_peaks)[keep]
    symbols = np.asarray(symbols)[keep]
    rr = rr_features(r_peaks, fs)

    beats, rr_out, labels = [], [], []
    for i, (r, sym) in enumerate(zip(r_peaks, symbols)):
        start, end = r - config.WINDOW_BEFORE, r + config.WINDOW_AFTER
        if start < 0 or end > len(signal) or np.isnan(rr[i]).any():
            continue
        seg = signal[start:end]
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        beats.append(seg.astype(np.float32))
        rr_out.append(rr[i].astype(np.float32))
        labels.append(config.CLASSES.index(config.AAMI_MAP[sym]))

    n = len(beats)
    return (np.array(beats).reshape(n, config.BEAT_LEN), np.array(rr_out).reshape(n, 4),
            np.array(labels, dtype=np.int64), np.full(n, patient))


def load_record(record, raw_dir=config.DATA_RAW):
    """Read one record: filtered MLII signal, R-peak sample indices, beat symbols."""
    import wfdb

    path = str(raw_dir / record)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    lead = rec.sig_name.index(config.LEAD)
    signal = bandpass(rec.p_signal[:, lead], rec.fs)
    return signal, ann.sample, ann.symbol


def build_dataset(records=None, raw_dir=config.DATA_RAW, out=config.BEATS_FILE):
    records = records or (config.DS1 + config.DS2)
    parts = []
    for rec in records:
        signal, peaks, symbols = load_record(rec, raw_dir)
        parts.append(extract_beats(signal, peaks, symbols, rec))
        print(f"  record {rec}: {len(parts[-1][2]):>5} beats")

    X, rr, y, patient = (np.concatenate(p) for p in zip(*parts))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, rr=rr, y=y, patient=patient)
    print(f"Saved {len(y)} beats to {out}")
    return X, rr, y, patient


def load_dataset(path=config.BEATS_FILE):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data` first.")
    d = np.load(path)
    return d["X"], d["rr"], d["y"], d["patient"]


def print_class_counts(y, patient):
    from src.split import patient_split

    masks = patient_split(patient)
    print(f"{'split':<6}" + "".join(f"{c:>8}" for c in config.CLASSES) + f"{'total':>9}")
    for name, m in masks.items():
        counts = np.bincount(y[m], minlength=len(config.CLASSES))
        print(f"{name:<6}" + "".join(f"{c:>8}" for c in counts) + f"{m.sum():>9}")


if __name__ == "__main__":
    download_mitdb()
    _, _, y, patient = build_dataset()
    print_class_counts(y, patient)
