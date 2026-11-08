"""Unified buffer split shared by both datasets.

The test block is a contiguous square of scan positions. Training and validation
use every position whose object patch shares no pixel with the test block;
the positions in the buffer ring around the block are not used. A fixed random
draw (RandomState(0)) of the pool forms the validation set, which selects the
checkpoint epoch. Amplitude labels are min-max normalised with the training
range; calibration statistics come from training patterns only.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class ReconstructionTestSet:
    """Diffraction inputs and physical labels of one square test region (row-major scan order)."""

    diffraction: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    grid_size: int


@dataclass(frozen=True)
class PreparedData:
    train_dataset: TensorDataset          # (counts, normalised amplitude, phase in rad), each (N, 1, H, W)
    validation_dataset: TensorDataset
    test_set: ReconstructionTestSet
    amplitude_range: tuple
    phase_range: tuple


def buffer_split(pool_mask, validation_size):
    """Sorted training and validation scan indices of a boolean pool mask."""

    ids = np.flatnonzero(pool_mask.ravel())
    validation = np.sort(np.random.RandomState(0).choice(ids, validation_size, replace=False))
    return np.setdiff1d(ids, validation), validation


def make_tensors(diffraction, amplitude, phase, amplitude_range):
    low, high = amplitude_range
    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(diffraction[:, None], dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(((amplitude - low) / (high - low))[:, None], dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(phase[:, None], dtype=np.float32)),
    )


def prepare(get, pool_mask, validation_size, test_ids, grid_size):
    """get(ids) -> (diffraction, amplitude, phase), float32 (len(ids), H, W), ids in row-major scan order."""

    train, validation = buffer_split(pool_mask, validation_size)
    x, a, p = get(train)
    amplitude_range = (float(a.min()), float(a.max()))
    phase_range = (float(p.min()), float(p.max()))
    return PreparedData(
        train_dataset=make_tensors(x, a, p, amplitude_range),
        validation_dataset=make_tensors(*get(validation), amplitude_range),
        test_set=ReconstructionTestSet(*get(test_ids), grid_size),
        amplitude_range=amplitude_range,
        phase_range=phase_range,
    )
