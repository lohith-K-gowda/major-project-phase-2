# -*- coding: utf-8 -*-
"""
loss.py - Standard Supervised Medical-Image Reconstruction Loss
L_total = L1_masked + lambda_ssim * L_SSIM
Default: lambda_ssim = 0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity (SSIM) Loss.
    Supports data range of 2.0 (for normalized [-1.0, 1.0] domain).
    """
    def __init__(self, window_size=11, sigma=1.5, data_range=2.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        
        # 1D Gaussian kernel
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        # 2D Gaussian window [1, 1, K, K]
        _2d = torch.outer(g, g).unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", _2d)

    def forward(self, img1, img2, mask=None):
        w = self.window.to(dtype=img1.dtype, device=img1.device)
        pad = self.window_size // 2
        
        mu1 = F.conv2d(img1, w, padding=pad)
        mu2 = F.conv2d(img2, w, padding=pad)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, w, padding=pad) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, w, padding=pad) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, w, padding=pad) - mu1_mu2
        
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
        )
        
        if mask is not None:
            # Masked SSIM loss inside anatomical body contour
            m = mask.to(dtype=img1.dtype, device=img1.device)
            m_sum = m.sum() + 1e-8
            return ((1.0 - ssim_map) * m).sum() / m_sum
        else:
            return 1.0 - ssim_map.mean()

class ResUNetLoss(nn.Module):
    """
    Standard supervised medical image translation loss for ResUNet baseline.
    L_total = L1_masked + lambda_ssim * L_SSIM_masked
    """
    def __init__(self, lambda_ssim=0.10, window_size=11):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.ssim_fn = SSIMLoss(window_size=window_size, data_range=2.0)

    def forward(self, pred, target, mask=None):
        """
        pred:   [B, 1, H, W] in [-1.0, 1.0]
        target: [B, 1, H, W] in [-1.0, 1.0]
        mask:   [B, 1, H, W] binary {0.0, 1.0} derived strictly from MRI
        """
        diff = torch.abs(pred - target)
        
        if mask is not None:
            mask_sum = mask.sum() + 1e-8
            l1_loss = (diff * mask).sum() / mask_sum
            l_ssim  = self.ssim_fn(pred, target, mask=mask)
        else:
            l1_loss = diff.mean()
            l_ssim  = self.ssim_fn(pred, target)
            
        total_loss = l1_loss + self.lambda_ssim * l_ssim
        
        return total_loss, {
            "total_loss": total_loss.item(),
            "l1_loss": l1_loss.item(),
            "ssim_loss": l_ssim.item()
        }

if __name__ == "__main__":
    crit = ResUNetLoss(lambda_ssim=0.1)
    p = torch.randn(2, 1, 128, 128).clamp(-1, 1)
    t = torch.randn(2, 1, 128, 128).clamp(-1, 1)
    m = (torch.rand(2, 1, 128, 128) > 0.3).float()
    loss, logs = crit(p, t, m)
    print("Loss test passed:", logs)
