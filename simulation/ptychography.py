"""Forward simulation of the AD_LTEM-derived ptychography dataset."""

from pathlib import Path

import numpy as np
from numpy.fft import fft2, fftshift
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.measure import block_reduce
from skimage.transform import resize
from tqdm import tqdm


def load_and_preprocess_image(image_path, new_size):
    image = imread(image_path).astype(np.float32)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = rgb2gray(image)
    return resize(image, new_size)


def adjust_amplitude_phase(amplitude, phase):
    """Map the chart to amplitude [0.5, 1] and a phase range of pi/3."""

    amplitude_adjusted = (
        0.5
        * (amplitude - amplitude.min())
        / (amplitude.max() - amplitude.min())
        + 0.5
    )
    phase_adjusted = -np.pi / 3 * (phase + np.pi) / (2 * np.pi)
    return amplitude_adjusted, phase_adjusted


def load_and_scale_probe(probe_path, block_size=(4, 4)):
    """Load the probe and block-average it to the patch size."""

    probe = np.load(probe_path).astype(np.complex64)
    scaled_real = block_reduce(probe.real, block_size=block_size, func=np.mean)
    scaled_imag = block_reduce(probe.imag, block_size=block_size, func=np.mean)
    return probe, scaled_real + 1j * scaled_imag


def calculate_step_size(probe_size, overlap_rate):
    return int(probe_size * (1 - overlap_rate / 100))


def simulate_pattern(probe, amplitude, phase, position):
    """Far-field intensity of one scan position and its object patch."""

    y, x = position
    amp_section = amplitude[y : y + probe.shape[0], x : x + probe.shape[1]]
    phase_section = phase[y : y + probe.shape[0], x : x + probe.shape[1]]
    if amp_section.shape != probe.shape:
        raise ValueError(
            f"Object section shape {amp_section.shape} does not match "
            f"probe shape {probe.shape}."
        )
    exit_wave = amp_section * np.exp(1j * phase_section) * probe
    diffraction = np.abs(fftshift(fft2(exit_wave))) ** 2
    return diffraction, amp_section, phase_section


def add_measurement_noise(diffraction, noise_level=0.01, rng=None):
    """Gaussian read noise followed by Poisson counting noise."""

    rng = np.random.default_rng() if rng is None else rng
    noisy = diffraction + rng.normal(0, noise_level, diffraction.shape)
    return rng.poisson(np.clip(noisy, 0, None)).astype(np.float32)


def generate_ad_ltem_dataset(
    amplitude_path,
    phase_path,
    probe_path,
    output_dir=".",
    image_size=(512, 512),
    overlap_rate=75,
    probe_block_size=(4, 4),
    noise_level=0.01,
    seed=None,
    paper_counts=None,
):
    """Simulate the scan and save labels and noisy patterns.

    Writes <overlap>_amp.npy and <overlap>_ph.npy (object patches) and
    <overlap>_diff.npy (|F{P O}|^2 with Gaussian plus Poisson noise). The labels
    are deterministic. With seed=None and the paper geometry, the noisy patterns are
    the recorded noise realization of the paper (paper_counts); an integer seed
    draws a new realization.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    amplitude = load_and_preprocess_image(amplitude_path, image_size)
    phase = load_and_preprocess_image(phase_path, image_size)
    amplitude, phase = adjust_amplitude_phase(amplitude, phase)
    _, probe = load_and_scale_probe(probe_path, block_size=probe_block_size)

    step_size = calculate_step_size(probe.shape[0], overlap_rate)
    if step_size <= 0:
        raise ValueError("The overlap rate must produce a positive scan step.")
    positions = [
        (y, x)
        for y in range(0, amplitude.shape[0] - probe.shape[0] + 1, step_size)
        for x in range(0, amplitude.shape[1] - probe.shape[1] + 1, step_size)
    ]
    simulated = [
        simulate_pattern(probe, amplitude, phase, position)
        for position in tqdm(positions, desc="Simulating scan positions")
    ]
    side = int(round(np.sqrt(len(positions))))
    shape = (side, side) + probe.shape
    diffraction, amp_sections, phase_sections = (
        np.asarray(values).reshape(shape) for values in zip(*simulated)
    )
    if seed is None and paper_counts is not None and Path(paper_counts).exists():
        with np.load(paper_counts) as f:
            noisy = f["counts"].astype(np.float32)
        if noisy.shape != diffraction.shape:
            raise ValueError("The recorded noise realization needs the paper geometry (75% overlap, 512 x 512 object).")
        print("Noisy patterns: recorded noise realization of the paper")
    else:
        noisy = add_measurement_noise(
            diffraction, noise_level, np.random.default_rng(0 if seed is None else seed)
        )
        print(f"Noisy patterns: new noise realization (seed {0 if seed is None else seed})")

    prefix = f"{overlap_rate:g}"
    arrays = {
        f"{prefix}_amp.npy": amp_sections,
        f"{prefix}_ph.npy": phase_sections,
        f"{prefix}_diff.npy": noisy,
    }
    for name, array in arrays.items():
        np.save(output_dir / name, array)
    print(
        f"Saved {side} x {side} scan positions ({probe.shape[0]} x "
        f"{probe.shape[1]} patterns, step {step_size}) to {output_dir}"
    )
    print(
        f"Amplitude [{amp_sections.min():.3f}, {amp_sections.max():.3f}], "
        f"phase [{phase_sections.min():.3f}, {phase_sections.max():.3f}] rad, "
        f"noisy counts [0, {noisy.max():.0f}]"
    )
    return arrays
