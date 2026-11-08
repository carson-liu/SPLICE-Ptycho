"""Evaluation protocol of the paper (patch-level and stitched-field metrics).

Patch level (both datasets): labels and predictions are min-max normalised
with the training label range; SSIM and PSNR use the range of the normalised
test labels.
Stitched field, AD_LTEM: the normalised patches are stitched by radially
weighted averaging; metrics use the range of the stitched ground truth.
Stitched field, experimental: the physical amplitude and phase are stitched
from the central 18 x 18 pixels of each patch and resized to 60 x 60; metrics
use the range of the stitched iterative reference.
The predicted phase is atan2(s~, c~) in (-pi, pi].
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def reconstruction_metrics(target, prediction, data_range=None):
    """Return MSE, MAE, SSIM, and PSNR for two images.

    data_range defaults to the range of the target.
    """

    if target.shape != prediction.shape:
        raise ValueError(
            f"Target and prediction shapes differ: {target.shape} != {prediction.shape}"
        )
    if data_range is None:
        data_range = float(target.max() - target.min()) or 1.0
    return {
        "MSE": float(np.mean((target - prediction) ** 2)),
        "MAE": float(np.mean(np.abs(target - prediction))),
        "SSIM": float(
            structural_similarity(target, prediction, data_range=data_range)
        ),
        "PSNR": float(
            peak_signal_noise_ratio(target, prediction, data_range=data_range)
        ),
    }


def normalized_patch_metrics(targets, predictions, value_range):
    """Per-patch metrics after min-max normalisation with the dataset range.

    SSIM and PSNR use the range of the test labels of the whole test set.
    """

    low, high = value_range
    targets = (np.asarray(targets, dtype=float) - low) / (high - low)
    predictions = (np.asarray(predictions, dtype=float) - low) / (high - low)
    label_range = float(targets.max() - targets.min())
    return [
        reconstruction_metrics(target, prediction, label_range)
        for target, prediction in zip(targets, predictions)
    ]


def average_patch_metrics(per_patch):
    """Return mean and standard deviation of metrics across all patches."""

    return {
        key: {
            "mean": float(np.mean([metrics[key] for metrics in per_patch])),
            "std": float(np.std([metrics[key] for metrics in per_patch])),
        }
        for key in per_patch[0]
    }


def evaluate_reconstruction(
    test_set,
    amp_pred,
    phase_pred,
    amplitude_range,
    phase_range,
    stitcher,
    normalize=True,
):
    """Patch and stitched metrics of one reconstruction of a test region.

    Returns a dictionary with per-patch metric summaries (normalised data
    only), stitched metrics, and the stitched physical images for plotting.
    """

    amp_true = np.asarray(test_set.amplitude, dtype=float)
    phase_true = np.asarray(test_set.phase, dtype=float)
    grid = test_set.grid_size
    images = {
        "amplitude_true": stitcher.stitch(amp_true, grid),
        "amplitude_pred": stitcher.stitch(amp_pred, grid),
        "phase_true": stitcher.stitch(phase_true, grid),
        "phase_pred": stitcher.stitch(phase_pred, grid),
    }
    result = {"patch": None, "images": images}
    if not normalize:
        result["stitched"] = {
            "amplitude": reconstruction_metrics(
                images["amplitude_true"], images["amplitude_pred"]
            ),
            "phase": reconstruction_metrics(
                images["phase_true"], images["phase_pred"]
            ),
        }
        return result

    result["patch"] = {
        "amplitude": average_patch_metrics(
            normalized_patch_metrics(amp_true, amp_pred, amplitude_range)
        ),
        "phase": average_patch_metrics(
            normalized_patch_metrics(phase_true, phase_pred, phase_range)
        ),
    }
    stitched = {}
    for name, true, pred, (low, high) in (
        ("amplitude", amp_true, amp_pred, amplitude_range),
        ("phase", phase_true, phase_pred, phase_range),
    ):
        stitched_true = stitcher.stitch((true - low) / (high - low), grid)
        stitched_pred = stitcher.stitch((pred - low) / (high - low), grid)
        stitched[name] = reconstruction_metrics(stitched_true, stitched_pred)
    result["stitched"] = stitched
    return result


def format_metrics(metrics, multiline=False):
    """Format a metric dictionary for terminal or figure output."""

    separator = "\n" if multiline else " | "
    return separator.join(
        [
            f"MSE {metrics['MSE']:.3e}   MAE {metrics['MAE']:.3e}",
            f"SSIM {metrics['SSIM']:.4f}   PSNR {metrics['PSNR']:.2f} dB",
        ]
    )


def format_average_metrics(summary):
    """Format mean +- standard deviation patch metrics."""

    return " | ".join(
        [
            f"MSE {summary['MSE']['mean']:.3e} +- {summary['MSE']['std']:.3e}",
            f"MAE {summary['MAE']['mean']:.3e} +- {summary['MAE']['std']:.3e}",
            f"SSIM {summary['SSIM']['mean']:.4f} +- {summary['SSIM']['std']:.4f}",
            f"PSNR {summary['PSNR']['mean']:.2f} +- {summary['PSNR']['std']:.2f} dB",
        ]
    )


def evaluate(dataset, test_set, amp_pred, phase_pred, amplitude_range, phase_range):
    """Patch and stitched metrics of one reconstruction of the test region of `dataset`."""

    from .stitching import stitcher_for

    if dataset == "ad_ltem":
        return evaluate_reconstruction(test_set, amp_pred, phase_pred, amplitude_range, phase_range, stitcher_for(dataset))
    result = evaluate_reconstruction(test_set, amp_pred, phase_pred, amplitude_range, phase_range,
                                     stitcher_for(dataset), normalize=False)
    result["patch"] = {
        name: average_patch_metrics(normalized_patch_metrics(true, pred, rng))
        for name, true, pred, rng in (("amplitude", test_set.amplitude, amp_pred, amplitude_range),
                                      ("phase", test_set.phase, phase_pred, phase_range))
    }
    return result
