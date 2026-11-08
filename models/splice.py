"""SPLICE: Saturation-aware Partitioned Lag-Intensity Correlation Encoding.

SPLICE turns one measured diffraction pattern I(q) (detector counts) into nine
views for the encoder and three detector signals for FiLM conditioning:

    M_k(q)       three soft radial virtual detectors (central, inner, outer band),
                 initialised at the radii enclosing 90 % and 99 % of the energy of
                 the mean training pattern; boundaries and edge width are learnable
    V_k          band-resolved intensity views  I_sat * kappa(g_k I M_k / I_sat)
    L_k^+, L_k^-  even and odd lag views, the real and imaginary parts of
                 Gamma_k = F^-1{I M_k}, the band-limited autocorrelation of the
                 exit wave; the odd part keeps the Friedel asymmetry
                 (I(q) - I(-q)) / 2 at every lag
    s_k          integrated detector signals log(1 + sum_q I M_k)

kappa(u) = u / (1 + u^4)^(1/4) is the smooth saturation. The gains g_k start at
values calibrated from the training patterns (99.5th percentile of each band
mapped to 0.85 I_sat) and carry a learnable factor in [1/2, 2]. No probe and no
labels are used.
"""

import math

import numpy as np
import torch
from torch import nn


SATURATION = 4095.0           # I_sat: range of a 12-bit detector
HEADROOM = 0.85               # calibrated percentile -> HEADROOM * I_sat
PERCENTILE = 0.995


def smooth_saturation(u, p=4.0):
    """kappa(u) = u / (1 + u^p)^(1/p): linear for small u, bounded by 1."""

    log_u = torch.log(u.clamp_min(1e-20))
    result = torch.exp(-nn.functional.softplus(-p * log_u) / p)
    return torch.where(u > 0, result, torch.zeros_like(result))


def energy_radius(patterns, energy=0.9):
    """Smallest radius (0.5-pixel steps) that encloses `energy` of the mean pattern."""

    mean = np.asarray(patterns, dtype=np.float32).mean(0)
    y, x = np.indices(mean.shape)
    radius_map = np.hypot(y - mean.shape[0] // 2, x - mean.shape[1] // 2)
    for radius in np.arange(0.5, mean.shape[0] / 2, 0.5):
        if mean[radius_map <= radius].sum() >= energy * mean.sum():
            return float(radius)
    return float(mean.shape[0] / 2)


def lag_transform(z, odd=False):
    """Real (even) or imaginary (odd) part of the centred inverse FFT over detector coordinates."""

    y = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(z, dim=(-2, -1))), dim=(-2, -1))
    return y.imag if odd else y.real


def dose_thinning(x, low=0.1):
    """Training-time dose thinning of counting data.

    Each pattern is thinned to a dose p ~ U(low, 1) (binomial thinning, normal
    approximation) and rescaled by 1 / p: the mean is unchanged and the counting
    noise is that of the lower dose.
    """

    p = low + (1 - low) * torch.rand(len(x), 1, 1, 1, device=x.device)
    m = x * p
    return (m + (m * (1 - p)).clamp_min(0).sqrt() * torch.randn_like(x)).round().clamp_min(0) / p


class SPLICE(nn.Module):
    """Learnable dual-domain encoding of a single diffraction pattern.

    forward(I) returns the nine views (V_1..3, L_1..3^+, L_1..3^-), each (N, 1, H, W);
    detector_signals(I) returns s(I), (N, 3).
    """

    num_bands = 3
    num_views = 9

    def __init__(self, patch_size):
        super().__init__()
        h = w = patch_size
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        self.register_buffer("radius", ((y - h // 2).square() + (x - w // 2).square()).float().sqrt())
        # calibrated from training patterns (SPLICE.calibrate or a checkpoint)
        self.register_buffer("initial_boundaries", torch.zeros(self.num_bands - 1))
        self.register_buffer("initial_edge_width", torch.tensor(0.5))
        self.register_buffer("initial_gains", torch.ones(self.num_bands))
        self.register_buffer("even_lag_scale", torch.ones(self.num_bands))
        self.register_buffer("odd_lag_scale", torch.ones(self.num_bands))
        # learnable, zero-initialised: log offsets of boundaries, edge width, and band gains
        self.boundary_offset = nn.Parameter(torch.zeros(self.num_bands - 1))
        self.edge_width_offset = nn.Parameter(torch.zeros(()))
        self.gain_offset = nn.Parameter(torch.zeros(self.num_bands))

    # ------------------------------------------------------------ detectors
    def masks(self):
        """Soft radial masks M_1..3 (central disc, inner annulus, outer region)."""

        boundaries = self.initial_boundaries * torch.exp(0.5 * torch.tanh(self.boundary_offset))
        width = self.initial_edge_width * torch.exp(torch.tanh(self.edge_width_offset))
        edges = [torch.sigmoid((self.radius - b) / width) for b in boundaries]
        out, inner = [], torch.ones_like(self.radius)
        for edge in edges:
            out.append(inner * (1 - edge))
            inner = edge
        out.append(inner)
        return out

    def relative_gains(self):
        return torch.exp(math.log(2) * torch.tanh(self.gain_offset))

    def intensity_views(self, x, masks=None):
        masks = self.masks() if masks is None else masks
        g = self.initial_gains * self.relative_gains()
        return [SATURATION * smooth_saturation(g[k] * x * m / SATURATION) for k, m in enumerate(masks)]

    def lag_views(self, x, odd, masks=None):
        masks = self.masks() if masks is None else masks
        g = self.relative_gains()
        scale = self.odd_lag_scale if odd else self.even_lag_scale
        views = []
        for k, m in enumerate(masks):
            a = g[k] * lag_transform(x * m, odd) / scale[k]
            views.append(HEADROOM * SATURATION * a.sign() * smooth_saturation(a.abs()))
        return views

    def forward(self, x):
        masks = self.masks()
        return (*self.intensity_views(x, masks), *self.lag_views(x, False, masks), *self.lag_views(x, True, masks))

    def detector_signals(self, x):
        """Integrated virtual-detector signals s_k = log(1 + sum_q I M_k)."""

        return torch.stack([torch.log1p((x * m).sum((1, 2, 3))) for m in self.masks()], 1)

    # ------------------------------------------------------------ calibration
    @torch.no_grad()
    def calibrate(self, train_patterns, disc_radius=None):
        """Set the calibrated buffers from training patterns (N, 1, H, W) only."""

        train_patterns = torch.as_tensor(train_patterns, dtype=torch.float32)
        if disc_radius is None:
            disc_radius = energy_radius(train_patterns[:, 0].numpy(), 0.9)
        r = self.radius
        mean = train_patterns[:, 0].double().mean(0)
        order = torch.argsort(r.flatten())
        cumulative = torch.cumsum(mean.flatten()[order], 0) / mean.sum()
        r99 = float(r.flatten()[order][torch.searchsorted(cumulative, torch.tensor(0.99, dtype=cumulative.dtype))])
        self.initial_boundaries.copy_(torch.tensor([max(disc_radius, 1.0), max(r99, disc_radius + 2)]))
        masks = self.masks()
        gains = [HEADROOM * SATURATION / torch.quantile((train_patterns[:, 0] * m)[:, m > 0.5].flatten()[::7].float(),
                                                        PERCENTILE).clamp_min(1e-3) for m in masks]
        self.initial_gains.copy_(torch.stack(gains))
        sub = train_patterns[:: max(1, len(train_patterns) // 256)]
        for odd, buffer in ((False, self.even_lag_scale), (True, self.odd_lag_scale)):
            scale = [torch.quantile(lag_transform(sub[:, 0] * m, odd).abs().flatten()[::3].float(), PERCENTILE).clamp_min(1e-6)
                     for m in masks]
            buffer.copy_(torch.stack(scale))
        return self
