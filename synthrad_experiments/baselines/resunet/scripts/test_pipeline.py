# -*- coding: utf-8 -*-
"""
test_pipeline.py - Verification script to validate model, loss, metrics, and training step
Runs end-to-end forward, backward, evaluation, and metric computation on synthetic batch.
"""

import torch
import numpy as np
from model import ResUNet
from loss import ResUNetLoss
from metrics import evaluate_slice

def test_full_pipeline():
    print("1. Testing ResUNet Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNet(in_channels=1, out_channels=1, base_filters=64).to(device)
    
    x = torch.randn(4, 1, 128, 128, device=device).clamp(-1, 1)
    target = torch.randn(4, 1, 128, 128, device=device).clamp(-1, 1)
    mask = (torch.rand(4, 1, 128, 128, device=device) > 0.3).float()
    
    out = model(x)
    assert out.shape == target.shape, f"Shape mismatch: {out.shape} vs {target.shape}"
    print(f"   [PASS] Forward output shape: {out.shape}, value range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    
    print("2. Testing ResUNet Loss (Masked L1 + 0.1 * Masked SSIM)...")
    criterion = ResUNetLoss(lambda_ssim=0.10).to(device)
    loss, logs = criterion(out, target, mask=mask)
    print(f"   [PASS] Loss computed: {logs}")
    
    print("3. Testing Backward Pass & Optimization Step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("   [PASS] Backward step executed successfully.")
    
    print("4. Testing Clinical Metrics Computation...")
    p_np = out[0, 0].detach().cpu().numpy()
    g_np = target[0, 0].detach().cpu().numpy()
    m_np = mask[0, 0].detach().cpu().numpy()
    
    res = evaluate_slice(p_np, g_np, m_np)
    print("   [PASS] Metrics computed on test slice:")
    for k, v in res.items():
        print(f"      - {k:18s}: {v:.4f}")
        
    print("=" * 60)
    print("ALL PIPELINE COMPONENTS VERIFIED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_full_pipeline()
