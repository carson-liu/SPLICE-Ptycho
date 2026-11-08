"""Inference, training, evaluation, stitching, and plotting utilities."""

from .inference import disable_tf32, reconstruct_patches, select_device
from .metrics import evaluate, format_average_metrics, format_metrics
from .stitching import AdaptiveImageStitcher, ExperimentalImageStitcher, stitcher_for
from .training import fit, train_model
from .visualization import plot_patch_comparison, plot_splice_views, plot_stitched

__all__ = [
    "AdaptiveImageStitcher", "ExperimentalImageStitcher", "disable_tf32", "evaluate", "format_average_metrics",
    "format_metrics", "plot_patch_comparison", "plot_splice_views", "plot_stitched", "reconstruct_patches",
    "select_device", "stitcher_for", "fit", "train_model",
]
