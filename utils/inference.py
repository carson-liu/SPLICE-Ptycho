"""Device selection and batched single-pattern inference."""

import numpy as np
import torch


def select_device():
    """CUDA, Apple MPS, or CPU, in that order."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def disable_tf32():
    """Use exact float32 convolutions and matrix products (cuDNN otherwise uses TF32)."""

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


@torch.no_grad()
def reconstruct_patches(model, diffraction, amplitude_range, batch_size=64, device=None):
    """Reconstruct each pattern independently.

    diffraction: (N, H, W) counts. Returns physical amplitude and phase
    phi = atan2(s~, c~) in (-pi, pi], both float64 (N, H, W).
    """

    device = next(model.parameters()).device if device is None else device
    torch.manual_seed(900000)                  # fixed random state at inference
    model.eval()
    amplitude, phase = [], []
    for start in range(0, len(diffraction), batch_size):
        x = torch.as_tensor(np.asarray(diffraction[start:start + batch_size])[:, None], dtype=torch.float32, device=device)
        outputs = model(x)
        amplitude.append(outputs[0][:, 0].cpu().numpy())
        phase.append(model.phase(outputs)[:, 0].cpu().numpy())
    low, high = amplitude_range
    amplitude, phase = np.concatenate(amplitude), np.concatenate(phase)
    if not (np.isfinite(amplitude).all() and np.isfinite(phase).all()):
        raise FloatingPointError("Nonfinite predictions")
    return low + amplitude.astype(float) * (high - low), phase.astype(float)
