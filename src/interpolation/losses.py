from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:

        diff = pred - target

        return torch.mean(
            torch.sqrt(diff * diff + self.eps * self.eps)
        )


def _gaussian_window(
    window_size: int,
    sigma: float,
    device,
    dtype
) -> torch.Tensor:

    coords = torch.arange(
        window_size,
        dtype=dtype,
        device=device
    ) - window_size // 2

    g = torch.exp(
        -(coords ** 2) / (2 * sigma ** 2)
    )

    g = g / g.sum()

    window_2d = g.unsqueeze(0) * g.unsqueeze(1)

    return window_2d.unsqueeze(0).unsqueeze(0)


class SSIMLoss(nn.Module):

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0
    ):

        super().__init__()

        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range

        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:

        if pred.shape[1] != 1 or target.shape[1] != 1:
            raise ValueError(
                "SSIMLoss expects single-channel (B,1,H,W) tensors."
            )

        window = _gaussian_window(
            self.window_size,
            self.sigma,
            pred.device,
            pred.dtype
        )

        pad = self.window_size // 2

        mu_p = F.conv2d(
            pred,
            window,
            padding=pad
        )

        mu_t = F.conv2d(
            target,
            window,
            padding=pad
        )

        mu_p_sq = mu_p * mu_p
        mu_t_sq = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sigma_p_sq = (
            F.conv2d(pred * pred, window, padding=pad)
            - mu_p_sq
        )

        sigma_t_sq = (
            F.conv2d(target * target, window, padding=pad)
            - mu_t_sq
        )

        sigma_pt = (
            F.conv2d(pred * target, window, padding=pad)
            - mu_pt
        )

        ssim_map = (
            (2 * mu_pt + self.C1)
            * (2 * sigma_pt + self.C2)
        ) / (
            (mu_p_sq + mu_t_sq + self.C1)
            * (sigma_p_sq + sigma_t_sq + self.C2)
        )

        return 1.0 - ssim_map.mean()