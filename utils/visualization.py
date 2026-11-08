"""SPLICE views, patch comparisons, and stitched reconstructions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .metrics import format_metrics


def _save(figure, save_path):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")


def _signed_log(v):
    return np.sign(v) * np.log10(1 + np.abs(v))


@torch.no_grad()
def plot_splice_views(model, pattern, save_path=None):
    """Measured pattern and the nine SPLICE views (intensity, even lag, odd lag) of one diffraction pattern."""

    device = next(model.parameters()).device
    x = torch.as_tensor(np.asarray(pattern, dtype=np.float32))[None, None].to(device)
    views = [v[0, 0].cpu().numpy() for v in model.splice(x)]
    figure, axes = plt.subplots(3, 4, figsize=(11, 8.4), constrained_layout=True)
    names = ("central", "inner", "outer")
    rows = (("Intensity view $V_{%d}$ (%s)", views[0:3], "magma"),
            ("Even lag view $L_{%d}^{+}$ (%s)", views[3:6], "RdBu_r"),
            ("Odd lag view $L_{%d}^{-}$ (%s)", views[6:9], "RdBu_r"))
    shown = axes[0, 0].imshow(np.log10(np.asarray(pattern, dtype=float) + 1), cmap="magma")
    axes[0, 0].set_title("Measured $\\log_{10}(1+I)$")
    figure.colorbar(shown, ax=axes[0, 0], fraction=0.046)
    signals = model.splice.detector_signals(x)[0].cpu().numpy()
    axes[1, 0].bar(names, signals, color="#555555")
    axes[1, 0].set_title("Detector signals $s_k$ (FiLM)")
    axes[1, 0].set_ylim(signals.min() - 1, signals.max() + 1)
    axes[2, 0].axis("off")
    masks = [m.cpu().numpy() for m in model.splice.masks()]
    axes[2, 0].imshow(sum((k + 1) * m for k, m in enumerate(masks)), cmap="viridis")
    axes[2, 0].set_title("Soft detectors $M_1, M_2, M_3$")
    for r, (title, images, cmap) in enumerate(rows):
        for k, image in enumerate(images):
            ax = axes[r, k + 1]
            image = _signed_log(image)
            shown = ax.imshow(image, cmap=cmap)
            figure.colorbar(shown, ax=ax, fraction=0.046)
            ax.set_title(title % (k + 1, names[k]), fontsize=10)
    for ax in axes.ravel():
        if ax is not axes[1, 0]:
            ax.set_xticks([]), ax.set_yticks([])
    figure.suptitle("SPLICE views (signed $\\log_{10}$ scale for display)")
    _save(figure, save_path)
    return figure


def plot_patch_comparison(test_set, amp_pred, phase_pred, indices, save_path=None):
    """Diffraction, ground-truth and reconstructed amplitude and phase of selected test patches."""

    figure, axes = plt.subplots(len(indices), 5, figsize=(14, 2.9 * len(indices)), squeeze=False, constrained_layout=True)
    titles = ("Diffraction $\\log_{10}(1+I)$", "Amplitude (label)", "Amplitude (SPLICE)", "Phase (label)", "Phase (SPLICE)")
    for row, i in enumerate(indices):
        panels = ((np.log10(test_set.diffraction[i] + 1), "magma"), (test_set.amplitude[i], "gray"),
                  (amp_pred[i], "gray"), (test_set.phase[i], "viridis"), (phase_pred[i], "viridis"))
        for col, (image, cmap) in enumerate(panels):
            ax = axes[row, col]
            shown = ax.imshow(image, cmap=cmap)
            figure.colorbar(shown, ax=ax, fraction=0.046)
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
        axes[row, 0].set_ylabel(f"test patch {i}")
    _save(figure, save_path)
    return figure


def plot_stitched(images, metrics, save_path=None):
    """Stitched ground truth and reconstruction of the test field, amplitude and phase."""

    figure, axes = plt.subplots(2, 2, figsize=(8.4, 8), constrained_layout=True)
    for row, (name, cmap) in enumerate((("amplitude", "gray"), ("phase", "viridis"))):
        true, pred = images[f"{name}_true"], images[f"{name}_pred"]
        for col, (image, label) in enumerate(((true, "ground truth"), (pred, "SPLICE framework"))):
            shown = axes[row, col].imshow(image, cmap=cmap)
            figure.colorbar(shown, ax=axes[row, col], fraction=0.046)
            axes[row, col].set_title(f"Stitched {name}, {label}", fontsize=10)
            axes[row, col].set_xticks([]), axes[row, col].set_yticks([])
        axes[row, 1].set_xlabel(format_metrics(metrics[name], multiline=True), fontsize=9)
    _save(figure, save_path)
    return figure
