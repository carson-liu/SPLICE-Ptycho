"""Experimental X-ray ptychography (APS sector 26, tungsten test pattern, PtychoNN data).

161 x 161 scan, 64 x 64 patterns, scan step 3 object pixels. Labels are the
iterative reconstruction released with the data.
Test: scan rows >= 101 and columns < 60 (60 x 60 = 3,600 patterns).
Pool: all positions outside rows >= 80 and columns <= 80, i.e. every patch that
shares no object pixel with the test block (19,360 positions); 805 of them form
the validation set and 18,555 the training set.
"""

from pathlib import Path

import numpy as np

from .common import prepare

DIFFRACTION = "20191008_39_diff_reduced.npz"
OBJECT = "20191008_39_amp_pha_10nm_full.npy"
URL = "https://huggingface.co/datasets/mcherukara/PtychoNN_data/resolve/main/{}?download=true"
SCAN, GRID, VALIDATION = 161, 60, 805

CALIBRATION_STRIDE = 7             # SPLICE and FiLM statistics from every 7th training pattern


def load_arrays(root="data"):
    root = Path(root)
    missing = [name for name in (DIFFRACTION, OBJECT) if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing experimental files in {root}: {missing}. See 'Data sources' in README.md.")
    with np.load(root / DIFFRACTION) as f:
        diffraction = f["arr_0"].astype(np.float32).reshape(-1, 64, 64)
    obj = np.load(root / OBJECT, mmap_mode="r").reshape(-1, 64, 64)
    return diffraction, obj


def pool_mask(scan_stride=1):
    """scan_stride k > 1: only every k-th scan row and column of the pool (sparse sampling, scan step 3k px)."""

    rows, cols = np.indices((SCAN, SCAN))
    return ~((rows >= 80) & (cols <= 80)) & (rows % scan_stride == 0) & (cols % scan_stride == 0)


def test_ids():
    rows, cols = np.indices((SCAN, SCAN))
    return np.flatnonzero(((rows >= 101) & (cols < 60)).ravel())


def prepare_experimental(root="data", scan_stride=1):
    """scan_stride 2: the sparse training set of the HDR comparison and SPLICE ablation (4,677 training, 203 validation).
    The test block stays dense."""
    diffraction, obj = load_arrays(root)

    def get(ids):
        o = np.asarray(obj[ids])
        return diffraction[ids], np.abs(o).astype(np.float32), np.angle(o).astype(np.float32)

    mask = pool_mask(scan_stride)
    return prepare(get, mask, round(VALIDATION * mask.sum() / pool_mask().sum()), test_ids(), GRID)
