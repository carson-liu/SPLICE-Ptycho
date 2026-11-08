"""End-to-end reconstruction network built on SPLICE.

    I --SPLICE--> nine views --shared encoder E--> 1x1 fusion --FiLM(s(I))--> f
    f + skip(I) --> amplitude decoder  -> A_hat = sigmoid(.)
                --> cosine decoder     -> c_hat \
                --> sine decoder       -> s_hat  > unit-circle projection (c~, s~), phi_hat = atan2(s~, c~)

One pattern in, one object patch out; full fields are formed by stitching.
"""

from pathlib import Path

import torch
from torch import nn

from .splice import SPLICE, dose_thinning, energy_radius


NCONV = 32
ENCODER_WIDTHS = (57, 114, 4 * NCONV)       # nine stacked views, width matched to two standard encoders


def conv_encoder(in_channels, widths):
    """Three stride-2 5 x 5 convolutions with ReLU (H x W -> H/8 x W/8)."""

    layers, channels = [], in_channels
    for width in widths:
        layers += [nn.Conv2d(channels, width, 5, stride=2, padding=2), nn.ReLU()]
        channels = width
    return nn.Sequential(*layers)


class TaskDecoder(nn.Module):
    """Three upsampling blocks; the skip feature joins after block 1."""

    def __init__(self, channels=4 * NCONV, skip_channels=2 * NCONV, out_channels=1):
        super().__init__()
        half = channels // 2

        def conv(in_ch, out_ch):
            return nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)

        self.block1 = nn.Sequential(conv(channels, channels), nn.ReLU(), conv(channels, channels), nn.ReLU())
        self.block2 = nn.Sequential(conv(channels + skip_channels, half), nn.ReLU(), conv(half, half), nn.ReLU())
        self.block3 = nn.Sequential(conv(half, half), nn.ReLU(), conv(half, half), nn.ReLU())
        self.out = conv(half, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, features, skip):
        x = self.upsample(self.block1(features))
        x = torch.cat([x, skip], dim=1)
        x = self.upsample(self.block2(x))
        x = self.upsample(self.block3(x))
        return self.out(x)


class DetectorFiLM(nn.Module):
    """FiLM of the fused feature from the standardised detector signals s(I).

    gamma, beta = MLP(z), z = clip((s - mu) / sd, -5, 5); f' = f (1 + gamma) + beta.
    The last layer starts at zero, so training starts from the unconditioned network.
    """

    def __init__(self, num_signals, channels, hidden=16):
        super().__init__()
        self.register_buffer("mu", torch.zeros(num_signals))
        self.register_buffer("sd", torch.ones(num_signals))
        self.mlp = nn.Sequential(nn.Linear(num_signals, hidden), nn.ReLU(), nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @torch.no_grad()
    def calibrate(self, signals):
        signals = signals.double()
        self.mu.copy_(signals.mean(0).float())
        self.sd.copy_(signals.std(0).float().clamp_min(1e-6))

    def forward(self, features, signals):
        z = ((signals - self.mu) / self.sd).clamp(-5, 5)
        gamma, beta = self.mlp(z)[:, :, None, None].chunk(2, 1)
        return features * (1 + gamma) + beta


class SPLICENet(nn.Module):
    """SPLICE front end, shared encoder, detector FiLM, skip path, and three task-specific decoders."""

    def __init__(self, patch_size, film_seed=None, dose_low=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.dose_low = dose_low                   # training-time dose thinning, p ~ U(dose_low, 1); 1 disables it
        self.splice = SPLICE(patch_size)
        self.encoder = conv_encoder(SPLICE.num_views, ENCODER_WIDTHS)
        self.fusion = nn.Conv2d(4 * NCONV, 4 * NCONV, 1, bias=False)
        self.amplitude_decoder = TaskDecoder()
        self.cos_decoder = TaskDecoder()
        self.sin_decoder = TaskDecoder()
        self.skip = nn.Sequential(
            nn.Conv2d(1, NCONV, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(NCONV, 2 * NCONV, 3, stride=2, padding=1)
        )
        if film_seed is not None:
            torch.manual_seed(film_seed)
        self.film = DetectorFiLM(SPLICE.num_bands, 4 * NCONV)
        with torch.no_grad():                      # lag channels start closed: training starts from the intensity views
            self.encoder[0].weight[:, SPLICE.num_bands:] = 0

    def forward(self, intensity):
        """intensity (N, 1, H, W) counts -> amplitude in [0, 1] (normalised), cos, sin on the unit circle."""

        if self.training and self.dose_low < 1:
            intensity = dose_thinning(intensity, self.dose_low)
        features = self.fusion(self.encoder(torch.cat(self.splice(intensity), 1)))
        features = self.film(features, self.splice.detector_signals(intensity))
        skip = self.skip(intensity)
        amplitude = torch.sigmoid(self.amplitude_decoder(features, skip))
        c, s = self.cos_decoder(features, skip), self.sin_decoder(features, skip)
        norm = torch.sqrt(c ** 2 + s ** 2 + 1e-8)
        return amplitude, c / norm, s / norm

    @staticmethod
    def phase(outputs):
        _, c, s = outputs
        return torch.atan2(s, c)


@torch.no_grad()
def build_model(train_patterns, seed=0, device="cpu", calibration_stride=1, dose_low=0.1):
    """Create and calibrate SPLICENet from training patterns (N, 1, H, W); no labels are used.

    The central-disc radius (90 % energy) uses every training pattern; the remaining SPLICE
    statistics and the FiLM standardisation use every `calibration_stride`-th pattern.
    """

    train_patterns = torch.as_tensor(train_patterns, dtype=torch.float32)
    disc_radius = energy_radius(train_patterns[:, 0].numpy(), 0.9)
    subset = train_patterns[::calibration_stride]
    torch.manual_seed(seed + 10000)
    splice_template = SPLICE(train_patterns.shape[-1]).calibrate(subset, disc_radius)
    torch.manual_seed(seed)
    model = SPLICENet(train_patterns.shape[-1], film_seed=seed + 30000, dose_low=dose_low)
    model.splice.load_state_dict(splice_template.state_dict())
    model.to(device)
    signals = torch.cat([model.splice.detector_signals(b.to(device)).cpu() for b in subset.split(256)])
    model.film.calibrate(signals)
    return model.to(device)


def save_checkpoint(model, path, amplitude_range, phase_range, **info):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    patch_size=model.patch_size, amplitude_range=tuple(amplitude_range), phase_range=tuple(phase_range),
                    **info), path)


def load_checkpoint(path, device="cpu"):
    """Return (model in eval mode, checkpoint dict)."""

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = SPLICENet(checkpoint["patch_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint
