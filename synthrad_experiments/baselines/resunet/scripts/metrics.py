# -*- coding: utf-8 -*-
"""
metrics.py - Clinical Evaluation Metrics for SynthRAD2023 MRI -> CT
Metrics calculated:
- Normalized space: MAE, RMSE, PSNR, SSIM, Masked SSIM
- Hounsfield Units (HU): HU-MAE, Masked HU-MAE, HU-RMSE, HU-PSNR
- Bone Segmentation (> 300 HU): Bone Dice, Bone HD95 (mm/px)
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from scipy.spatial.distance import directed_hausdorff

HU_MIN = -1000.0
HU_MAX = 1500.0
HU_RANGE = HU_MAX - HU_MIN
BONE_THRESHOLD_HU = 300.0

def norm_to_hu(norm_arr):
    """
    Converts normalized [-1.0, 1.0] image array to Hounsfield Units [-1000.0, 1500.0].
    """
    unit = np.clip((norm_arr + 1.0) / 2.0, 0.0, 1.0)
    return unit * HU_RANGE + HU_MIN

def compute_bone_dice_and_hd95(real_bone_mask, fake_bone_mask):
    """
    Computes Bone Dice coefficient and 95th percentile Hausdorff Distance (HD95).
    real_bone_mask: boolean numpy array (H, W)
    fake_bone_mask: boolean numpy array (H, W)
    """
    intersection = np.sum(real_bone_mask & fake_bone_mask)
    total_area   = np.sum(real_bone_mask) + np.sum(fake_bone_mask)
    
    if total_area == 0:
        # Both masks empty -> perfect agreement on absence of bone
        return 1.0, 0.0
    
    dice = float((2.0 * intersection) / (total_area + 1e-8))
    
    coords_real = np.argwhere(real_bone_mask)
    coords_fake = np.argwhere(fake_bone_mask)
    
    if len(coords_real) == 0 or len(coords_fake) == 0:
        # One is empty and other is not -> maximum penalty HD95
        return float(dice), 50.0
    
    # Directed Hausdorff distances
    d_fwd = directed_hausdorff(coords_real, coords_fake)[0]
    d_rev = directed_hausdorff(coords_fake, coords_real)[0]
    hd95 = float(np.percentile([d_fwd, d_rev], 95))
    
    return float(dice), float(hd95)

def evaluate_slice(pred_norm, gt_norm, mask_arr=None):
    """
    Evaluates a single 2D slice [H, W] in [-1.0, 1.0].
    mask_arr: binary [H, W] {0.0, 1.0} derived from MRI
    """
    pred_c = np.clip(pred_norm, -1.0, 1.0)
    gt_c   = np.clip(gt_norm, -1.0, 1.0)
    
    # Unit domain [0, 1] for skimage
    pred_unit = (pred_c + 1.0) / 2.0
    gt_unit   = (gt_c + 1.0) / 2.0
    
    # 1. Normalized space metrics
    mae_norm = float(np.mean(np.abs(pred_unit - gt_unit)))
    rmse_norm = float(np.sqrt(np.mean((pred_unit - gt_unit) ** 2)))
    psnr_norm = float(compute_psnr(gt_unit, pred_unit, data_range=1.0))
    ssim_val, ssim_map = compute_ssim(gt_unit, pred_unit, data_range=1.0, full=True)
    ssim_norm = float(ssim_val)
    
    # 2. Masked metrics (if body mask available)
    if mask_arr is not None:
        body = mask_arr > 0.5
        if body.sum() > 0:
            masked_mae_norm = float(np.mean(np.abs(pred_unit[body] - gt_unit[body])))
            masked_ssim = float(np.mean(ssim_map[body]))
        else:
            masked_mae_norm = mae_norm
            masked_ssim = ssim_norm
    else:
        body = np.ones_like(pred_norm, dtype=bool)
        masked_mae_norm = mae_norm
        masked_ssim = ssim_norm
        
    # 3. HU domain metrics
    pred_hu = norm_to_hu(pred_c)
    gt_hu   = norm_to_hu(gt_c)
    
    hu_mae = float(np.mean(np.abs(pred_hu - gt_hu)))
    hu_rmse = float(np.sqrt(np.mean((pred_hu - gt_hu) ** 2)))
    
    if mask_arr is not None and body.sum() > 0:
        masked_hu_mae = float(np.mean(np.abs(pred_hu[body] - gt_hu[body])))
        masked_mse_hu = np.mean((pred_hu[body] - gt_hu[body]) ** 2)
        masked_hu_psnr = float(20.0 * np.log10(HU_RANGE / (np.sqrt(masked_mse_hu) + 1e-8)))
    else:
        masked_hu_mae = hu_mae
        masked_hu_psnr = float(20.0 * np.log10(HU_RANGE / (hu_rmse + 1e-8)))
        
    # 4. Bone segmentation (> 300 HU inside body mask)
    real_bone = (gt_hu > BONE_THRESHOLD_HU) & body
    fake_bone = (pred_hu > BONE_THRESHOLD_HU) & body
    bone_dice, bone_hd95 = compute_bone_dice_and_hd95(real_bone, fake_bone)
    
    return {
        "MAE_norm": mae_norm,
        "RMSE_norm": rmse_norm,
        "PSNR_dB": psnr_norm,
        "SSIM": ssim_norm,
        "Masked_SSIM": masked_ssim,
        "Masked_MAE_norm": masked_mae_norm,
        "HU_MAE": hu_mae,
        "Masked_HU_MAE": masked_hu_mae,
        "HU_RMSE": hu_rmse,
        "Masked_HU_PSNR": masked_hu_psnr,
        "Bone_Dice": bone_dice,
        "Bone_HD95_px": bone_hd95
    }

if __name__ == "__main__":
    p = np.random.uniform(-1, 1, (128, 128))
    g = np.random.uniform(-1, 1, (128, 128))
    m = (np.random.uniform(0, 1, (128, 128)) > 0.4).astype(np.float32)
    res = evaluate_slice(p, g, m)
    for k, v in res.items():
        print(f"{k}: {v:.4f}")
