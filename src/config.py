"""Central configuration: paths, signal constants, class mapping and patient splits."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "mitdb"
DATA_PROCESSED = ROOT / "data" / "processed"
BEATS_FILE = DATA_PROCESSED / "beats.npz"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SEED = 42

# --- Signal settings -------------------------------------------------------
FS = 360                 # MIT-BIH sampling rate (Hz)
LEAD = "MLII"            # modified limb lead II, present in every non-paced record
WINDOW_BEFORE = 90       # samples before the R-peak (~0.25 s)
WINDOW_AFTER = 160       # samples after the R-peak (~0.44 s)
BEAT_LEN = WINDOW_BEFORE + WINDOW_AFTER

# --- AAMI EC57 classes -----------------------------------------------------
# N: normal / bundle branch block, S: supraventricular ectopic,
# V: ventricular ectopic, F: fusion, Q: unknown / unclassifiable
CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

# --- Inter-patient split (de Chazal et al., 2004) ---------------------------
# The standard benchmark split: no patient appears in both train and test.
# Paced records (102, 104, 107, 217) are excluded, as AAMI recommends.
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]

# Patients carved out of DS1 for early stopping / model selection.
# Chosen because they contain S and V beats, so validation macro-F1 is meaningful.
VAL_PATIENTS = ["106", "220", "223"]
