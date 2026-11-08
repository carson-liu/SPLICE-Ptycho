"""Simulated AD_LTEM dataset: 61 x 61 scan, 32 x 32 patterns, scan step 8 px.

Test: the last 12 x 12 scan positions (rows and columns 49-60, 144 patterns).
Pool: all positions outside rows >= 46 and columns >= 46, i.e. every patch that
shares no object pixel with the test block (3,496 positions); 100 of them form
the validation set and 3,396 the training set.
"""

from pathlib import Path

import numpy as np

from .common import prepare

FILES = {"amplitude": "75_amp.npy", "phase": "75_ph.npy", "diffraction": "75_diff.npy"}
SCAN, GRID, BUFFER_START, VALIDATION = 61, 12, 46, 100

CALIBRATION_STRIDE = 1             # SPLICE and FiLM statistics from every training pattern


def load_arrays(root="data"):
    root = Path(root)
    missing = [name for name in FILES.values() if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing AD_LTEM files in {root}: {missing}. Run `python main.py --mode generate` first.")
    return {key: np.load(root / name).astype(np.float32).reshape(-1, 32, 32) for key, name in FILES.items()}


def pool_mask():
    rows, cols = np.indices((SCAN, SCAN))
    return ~((rows >= BUFFER_START) & (cols >= BUFFER_START))


def test_ids():
    rows, cols = np.indices((SCAN, SCAN))
    return np.flatnonzero(((rows >= SCAN - GRID) & (cols >= SCAN - GRID)).ravel())


def prepare_ad_ltem(root="data"):
    a = load_arrays(root)
    return prepare(lambda ids: (a["diffraction"][ids], a["amplitude"][ids], a["phase"][ids]), pool_mask(), VALIDATION, test_ids(), GRID)
