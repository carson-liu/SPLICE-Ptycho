"""Composite amplitude and circular-phase objective of the reconstruction network.

    L = l(A, A_hat) + w_p [l(cos phi, c~) + l(sin phi, s~)]
    l = MSE + lambda_g * gradient loss + lambda_s * (1 - SSIM)

with w_p = 1.3, lambda_g = 0.12 and lambda_s = 0.1 (the SSIM terms of cosine and
sine are averaged). After the unit-circle projection the cosine-sine MSE equals
2 [1 - cos(phi_hat - phi)], a branch-cut-free phase error, so no separate
circular or unit-norm term is needed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient_loss(prediction, target):
    """Mean absolute error of horizontal and vertical finite differences."""

    loss_x = F.l1_loss(
        prediction[..., :, 1:] - prediction[..., :, :-1],
        target[..., :, 1:] - target[..., :, :-1],
    )
    loss_y = F.l1_loss(
        prediction[..., 1:, :] - prediction[..., :-1, :],
        target[..., 1:, :] - target[..., :-1, :],
    )
    return (loss_x + loss_y) / 2.0


class SSIMLoss(nn.Module):
    """1 - SSIM with an 11 x 11 Gaussian window (sigma 1.5)."""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        gauss = torch.Tensor(
            [
                np.exp(-((x - window_size // 2) ** 2) / (2 * sigma**2))
                for x in range(window_size)
            ]
        )
        gauss = gauss / gauss.sum()
        kernel = (gauss.unsqueeze(1) * gauss.unsqueeze(0))[None, None]
        self.register_buffer("kernel", kernel, persistent=False)
        self.padding = window_size // 2

    def forward(self, prediction, target):
        c1, c2 = 0.01**2, 0.03**2
        channels = prediction.shape[1]
        kernel = self.kernel.repeat(channels, 1, 1, 1)

        def blur(x):
            return F.conv2d(x, kernel, padding=self.padding, groups=channels)

        mu_pred, mu_target = blur(prediction), blur(target)
        var_pred = blur(prediction * prediction) - mu_pred**2
        var_target = blur(target * target) - mu_target**2
        covariance = blur(prediction * target) - mu_pred * mu_target
        numerator = (2 * mu_pred * mu_target + c1) * (2 * covariance + c2)
        denominator = (mu_pred**2 + mu_target**2 + c1) * (
            var_pred + var_target + c2
        )
        return 1.0 - torch.mean(numerator / denominator)


class CompositeLoss(nn.Module):
    """Amplitude and projected cosine-sine phase objective (the training loss of the paper)."""

    def __init__(self, phase_weight=1.3, grad_weight=0.12, ssim_weight=0.1):
        super().__init__()
        self.phase_weight = phase_weight
        self.grad_weight = grad_weight
        self.ssim_weight = ssim_weight
        self.ssim_loss = SSIMLoss()

    def forward(self, outputs, amplitude, phase):
        pred_amp, pred_cos, pred_sin = outputs
        cos_phase, sin_phase = torch.cos(phase), torch.sin(phase)

        loss_amplitude = F.mse_loss(pred_amp, amplitude)
        loss_amplitude = loss_amplitude + self.grad_weight * gradient_loss(
            pred_amp, amplitude
        )
        loss_amplitude = loss_amplitude + self.ssim_weight * self.ssim_loss(
            pred_amp, amplitude
        )

        loss_phase = F.mse_loss(pred_cos, cos_phase) + F.mse_loss(
            pred_sin, sin_phase
        )
        loss_phase = loss_phase + self.grad_weight * (
            gradient_loss(pred_cos, cos_phase)
            + gradient_loss(pred_sin, sin_phase)
        )
        loss_phase = loss_phase + self.ssim_weight * (
            self.ssim_loss(pred_cos, cos_phase)
            + self.ssim_loss(pred_sin, sin_phase)
        ) / 2
        return loss_amplitude + self.phase_weight * loss_phase
