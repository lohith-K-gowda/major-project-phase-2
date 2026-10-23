# -*- coding: utf-8 -*-
"""
model.py - Supervised Residual U-Net (ResUNet) Baseline for SynthRAD2023 MRI -> CT
Architectural Specification:
- Input: 1-channel MRI [B, 1, H, W] in [-1.0, 1.0]
- Output: 1-channel synthetic CT [B, 1, H, W] in [-1.0, 1.0]
- Pure supervised CNN: No diffusion timesteps, no attention gates, no GAN discriminator.
- Encoder: 1 -> 64 -> 128 -> 256 -> 512
- Bottleneck: 512
- Decoder: 512 -> 256 -> 128 -> 64 -> 1
- Residual convolutional blocks + standard skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Two-stage residual convolution block with normalization and LeakyReLU activation.
    Includes 1x1 projection shortcut when channel dimension changes.
    """
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        # Determine valid group count for GroupNorm
        groups = num_groups if out_channels % num_groups == 0 else (4 if out_channels % 4 == 0 else 1)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1  = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2  = nn.LeakyReLU(0.2, inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(groups, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act2(h + residual)

class ResUNet(nn.Module):
    """
    Supervised Residual U-Net (ResUNet) baseline for paired MRI-to-CT translation.
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super().__init__()
        f = base_filters
        
        # Encoder
        self.enc1 = ResBlock(in_channels, f)        # 1 -> 64
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ResBlock(f, f * 2)               # 64 -> 128
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ResBlock(f * 2, f * 4)           # 128 -> 256
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ResBlock(f * 4, f * 8)           # 256 -> 512
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(f * 8, f * 8),                   # 512 -> 512
            ResBlock(f * 8, f * 8)
        )
        
        # Decoder with transposed convolutions + skip connections
        self.up4 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec4 = ResBlock(f * 8 + f * 4, f * 4)   # (512 + 256) -> 256
        
        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec3 = ResBlock(f * 4 + f * 2, f * 2)   # (256 + 128) -> 128
        
        self.up2 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec2 = ResBlock(f * 2 + f, f)           # (128 + 64) -> 64
        
        self.up1 = nn.ConvTranspose2d(f, f, kernel_size=2, stride=2)
        self.dec1 = ResBlock(f + f, f)               # (64 + 64) -> 64
        
        # Final projection to 1-channel synthetic CT with Tanh activation [-1.0, 1.0]
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)
        self.final_act = nn.Tanh()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                  # [B, 64, H, W]
        e2 = self.enc2(self.pool1(e1))     # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))     # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool3(e3))     # [B, 512, H/8, W/8]
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4)) # [B, 512, H/16, W/16]
        
        # Decoder with skip connections
        d4 = self.up4(b)                   # [B, 256, H/8, W/8]
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)                  # [B, 128, H/4, W/4]
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)                  # [B, 64, H/2, W/2]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)                  # [B, 64, H, W]
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        out = self.final_act(self.final_conv(d1))
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    net = ResUNet(in_channels=1, out_channels=1, base_filters=64)
    x = torch.randn(2, 1, 128, 128)
    y = net(x)
    print(f"ResUNet initialized successfully.")
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total trainable parameters: {count_parameters(net):,}")
