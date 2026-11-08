"""Generate data, train, or test the SPLICE reconstruction framework.

    python main.py --mode generate                                  # simulated AD_LTEM data -> data/
    python main.py --dataset ad_ltem --mode test                    # pretrained model
    python main.py --dataset experimental --mode test
    python main.py --dataset ad_ltem --mode train --seed 0
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets import ad_ltem, experimental, prepare_ad_ltem, prepare_experimental  # noqa: E402
from models import load_checkpoint  # noqa: E402
from simulation import generate_ad_ltem_dataset  # noqa: E402
from utils import (  # noqa: E402
    disable_tf32, evaluate, format_average_metrics, format_metrics, plot_patch_comparison, plot_stitched,
    reconstruct_patches, select_device, train_model,
)

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "simulation" / "assets"
DATASETS = {
    "ad_ltem": dict(prepare=prepare_ad_ltem, stride=ad_ltem.CALIBRATION_STRIDE, patches=(0, 77, 140)),
    "experimental": dict(prepare=prepare_experimental, stride=experimental.CALIBRATION_STRIDE, patches=(0, 1830, 3599)),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="ad_ltem")
    parser.add_argument("--mode", choices=("test", "train", "generate"), default="test")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=None,
                        help="checkpoint(s) to test; several are reported as mean +- std")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--device", default=None)
    parser.add_argument("--noise-seed", type=int, default=None,
                        help="generate: draw a new noise realization instead of the recorded one of the paper")
    return parser.parse_args()


def mean_std(values, digits):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():{digits}} +- {values.std(ddof=1):{digits}}"


def print_results(dataset, results):
    n = len(results)
    for scope, title in (("patch", "Per-patch metrics"), ("stitched", "Stitched-field metrics")):
        print(f"{title}, {dataset} test region" + (f" (mean +- std over {n} models)" if n > 1 else ""))
        for name in ("amplitude", "phase"):
            if n == 1:
                text = format_average_metrics(results[0][scope][name]) if scope == "patch" else format_metrics(results[0][scope][name])
            else:
                get = (lambda r, k: r[scope][name][k]["mean"]) if scope == "patch" else (lambda r, k: r[scope][name][k])
                text = " | ".join(f"{k} {mean_std([get(r, k) for r in results], {'SSIM': '.4f', 'PSNR': '.2f'}.get(k, '.3e'))}"
                                  for k in ("MSE", "MAE", "SSIM", "PSNR"))
            print(f"  {name.capitalize():10s} {text}")


def run_test(args, device):
    settings = DATASETS[args.dataset]
    checkpoints = args.checkpoint or [ROOT / "checkpoints" / f"{args.dataset}.pth"]
    out = args.output_dir or ROOT / "results" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    test_set = settings["prepare"](args.data_root).test_set
    results = []
    for path in checkpoints:
        model, ck = load_checkpoint(path, device)
        amp, phase = reconstruct_patches(model, test_set.diffraction, ck["amplitude_range"])
        results.append(evaluate(args.dataset, test_set, amp, phase, ck["amplitude_range"], ck["phase_range"]))
        print(f"{path}: SPLICENet")
        if len(checkpoints) == 1:
            images = results[0]["images"]
            np.savez_compressed(out / "reconstruction.npz", amplitude=amp, phase=phase,
                                stitched_amplitude=images["amplitude_pred"], stitched_phase=images["phase_pred"])
            plt.close(plot_patch_comparison(test_set, amp, phase, settings["patches"], out / "patch_comparison.png"))
            plt.close(plot_stitched(images, results[0]["stitched"], out / "stitched_reconstruction.png"))
    print_results(args.dataset, results)
    if len(checkpoints) == 1:
        print(f"Patch and stitched reconstructions saved to {out}")


def run_train(args, device):
    settings = DATASETS[args.dataset]
    data = settings["prepare"](args.data_root)
    checkpoint = args.checkpoint[0] if args.checkpoint else ROOT / "runs" / args.dataset / f"seed_{args.seed}" / "model.pth"
    model = train_model(data, checkpoint, seed=args.seed, epochs=args.epochs, device=device,
                        dataset_name=args.dataset, calibration_stride=settings["stride"])
    amp, phase = reconstruct_patches(model, data.test_set.diffraction, data.amplitude_range)
    print_results(args.dataset, [evaluate(args.dataset, data.test_set, amp, phase, data.amplitude_range, data.phase_range)])


def run_generate(args, device):
    if args.dataset != "ad_ltem":
        raise ValueError("Data generation is only available for AD_LTEM; see README.md for the experimental data.")
    generate_ad_ltem_dataset(ASSETS / "amplitude.tiff", ASSETS / "phase.tiff", ASSETS / "probe.npy", args.data_root,
                             seed=args.noise_seed, paper_counts=ASSETS / "ad_ltem_counts.npz")


def main():
    args = parse_args()
    disable_tf32()
    if args.mode == "train":
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device) if args.device else select_device()
    {"test": run_test, "train": run_train, "generate": run_generate}[args.mode](args, device)


if __name__ == "__main__":
    main()
