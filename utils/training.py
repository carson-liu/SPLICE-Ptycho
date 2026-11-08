"""Training protocol of the paper.

Adam (learning rate 1e-3) with a triangular-2 cyclic schedule between 1e-4 and
1e-3 (half cycle of six epochs), batch size 64, gradient-norm clipping at 1,
45 epochs, and the epoch with the lowest validation loss is kept.
"""

import copy
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models import CompositeLoss, build_model, save_checkpoint

EPOCHS, BATCH_SIZE, LEARNING_RATE = 45, 64, 1e-3


@torch.no_grad()
def validation_loss(model, loss_fn, dataset, device, batch_size=BATCH_SIZE):
    """Mean supervised loss in eval mode; the random state of training is not touched."""

    x, a, p = dataset.tensors
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(900000)
        model.eval()
        total = sum(float(loss_fn(model(x[i:i + batch_size].to(device)), a[i:i + batch_size].to(device),
                                  p[i:i + batch_size].to(device))) * len(x[i:i + batch_size])
                    for i in range(0, len(x), batch_size))
        model.train()
    return total / len(x)


def fit(model, data, seed=0, epochs=EPOCHS, device="cuda", loss_fn=None, log=print):
    """Train any (amplitude, phase-output) model with the paper's protocol; return (best_epoch, history)."""

    device = torch.device(device)
    loss_fn = (CompositeLoss() if loss_fn is None else loss_fn).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(data.train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed))
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LEARNING_RATE / 10, max_lr=LEARNING_RATE,
                                                  step_size_up=6 * (len(data.train_dataset) // BATCH_SIZE + 1),
                                                  cycle_momentum=False, mode="triangular2")
    log(f"{len(data.train_dataset)} training, {len(data.validation_dataset)} validation patterns | "
        f"parameters {sum(p.numel() for p in model.parameters()):,} | seed {seed} | device {device}")
    torch.manual_seed(seed + 20000)
    history, best, best_epoch, state = [], float("inf"), 0, None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, a, p in loader:
            x, a, p = x.to(device), a.to(device), p.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), a, p)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(x)
        val = validation_loss(model, loss_fn, data.validation_dataset, device)
        history.append(dict(epoch=epoch, train_loss=total / len(data.train_dataset), val_loss=val))
        if val < best:
            best, best_epoch, state = val, epoch, copy.deepcopy(model.state_dict())
        log(f"epoch {epoch:02d}/{epochs} | train {history[-1]['train_loss']:.5f} | validation {val:.5f}"
            f"{' *' if best_epoch == epoch else ''} | {time.perf_counter() - start:.0f}s")
    model.load_state_dict(state)
    model.eval()
    return best_epoch, history


def train_model(data, checkpoint_path, seed=0, epochs=EPOCHS, device="cuda", dataset_name="", calibration_stride=1, log=print):
    """Calibrate SPLICE on the training patterns, train SPLICENet, keep the best validation epoch."""

    model = build_model(data.train_dataset.tensors[0], seed=seed, device=torch.device(device), calibration_stride=calibration_stride)
    log(f"{dataset_name}: SPLICENet")
    best_epoch, history = fit(model, data, seed, epochs, device, log=log)
    save_checkpoint(model, checkpoint_path, data.amplitude_range, data.phase_range,
                    dataset=dataset_name, seed=seed, best_epoch=best_epoch, epochs=epochs)
    Path(checkpoint_path).with_name("history.json").write_text(json.dumps(history, indent=1))
    log(f"best validation epoch {best_epoch}; checkpoint {checkpoint_path}")
    return model
