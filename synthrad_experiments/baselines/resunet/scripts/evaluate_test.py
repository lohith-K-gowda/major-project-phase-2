# -*- coding: utf-8 -*-
"""
evaluate_test.py - Frozen Evaluation of ResUNet Baseline on Held-Out Test Patients
SynthRAD2023 Task 1 (Brain MRI -> CT)
Strict Scientific Integrity:
- Only evaluated on the 6 isolated test patients: 1BA054, 1BA097, 1BA172, 1BA234, 1BA256, 1BA307
- Generates:
  1. Slice-level metrics CSV
  2. Patient-level metrics CSV
  3. Aggregate test summary CSV
  4. Publication-ready comparison figures
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model import ResUNet
from dataset import SynthRADResUNetDataset, get_synthrad_loaders, find_dataset_root
from metrics import evaluate_slice, norm_to_hu, HU_MIN, HU_MAX, HU_RANGE

def run_test_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("=" * 80)
    print(" FROZEN RESUNET BASELINE EVALUATION (TEST PATIENTS) ")
    print("=" * 80)
    print(f"Device: {device}")
    
    exp_dir = args.exp_dir
    ckpt_path = args.checkpoint or os.path.join(exp_dir, "checkpoints", "best_model.pth")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
    print(f"Loading Checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Initialize model
    model = ResUNet(in_channels=1, out_channels=1, base_filters=args.base_filters).to(device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model weights loaded successfully. ResUNet is FROZEN.")
    
    # Locate dataset
    data_root = args.data_dir or find_dataset_root()
    if not data_root or not os.path.exists(data_root):
        raise FileNotFoundError(f"Dataset root not found at: {args.data_dir}")
        
    # Get test data loader
    _, _, test_loader, split_info = get_synthrad_loaders(
        base_dir=data_root,
        split_json_path=args.split_json,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers
    )
    
    test_patients = split_info["test_patients"]
    print(f"Test Patients ({len(test_patients)}): {test_patients}")
    print(f"Test Slices   : {split_info['test_slices']}")
    
    # Per-slice evaluation loop
    slice_records = []
    patient_slices = {p: [] for p in test_patients}
    
    qualitative_samples = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Held-Out Test Set"):
            mri  = batch["mri"].to(device)
            ct   = batch["ct"].to(device)
            mask = batch["mask"].to(device)
            
            pids = batch["patient_id"]
            names = batch["slice_name"]
            
            # Pure inference pass (supervision only)
            pred = model(mri)
            
            # Enforce exterior mask background
            pred = torch.where(mask > 0.5, pred, torch.full_like(pred, -1.0))
            
            p_np = pred.squeeze(1).cpu().numpy()
            g_np = ct.squeeze(1).cpu().numpy()
            m_np = mask.squeeze(1).cpu().numpy()
            i_np = mri.squeeze(1).cpu().numpy()
            
            for idx in range(mri.size(0)):
                pid = pids[idx]
                sname = names[idx]
                
                res = evaluate_slice(p_np[idx], g_np[idx], m_np[idx])
                res["patient_id"] = pid
                res["slice_name"] = sname
                
                slice_records.append(res)
                patient_slices[pid].append((res["HU_MAE"], i_np[idx], g_np[idx], p_np[idx], sname))
                
    df_slices = pd.DataFrame(slice_records)
    
    # 1. Save Slice-level Metrics
    slice_csv_path = os.path.join(exp_dir, "metrics", "test_slice_metrics.csv")
    df_slices.to_csv(slice_csv_path, index=False)
    print(f"✅ Saved slice metrics to: {slice_csv_path}")
    
    # 2. Patient-level Aggregations
    patient_agg = []
    for pid in test_patients:
        p_df = df_slices[df_slices["patient_id"] == pid]
        patient_agg.append({
            "patient_id": pid,
            "slice_count": len(p_df),
            "MAE_norm": f"{p_df['MAE_norm'].mean():.4f} ± {p_df['MAE_norm'].std():.4f}",
            "PSNR_dB": f"{p_df['PSNR_dB'].mean():.2f} ± {p_df['PSNR_dB'].std():.2f}",
            "SSIM": f"{p_df['SSIM'].mean():.4f} ± {p_df['SSIM'].std():.4f}",
            "Masked_SSIM": f"{p_df['Masked_SSIM'].mean():.4f} ± {p_df['Masked_SSIM'].std():.4f}",
            "HU_MAE": f"{p_df['HU_MAE'].mean():.1f} ± {p_df['HU_MAE'].std():.1f}",
            "Masked_HU_MAE": f"{p_df['Masked_HU_MAE'].mean():.1f} ± {p_df['Masked_HU_MAE'].std():.1f}",
            "Bone_Dice": f"{p_df['Bone_Dice'].mean():.4f} ± {p_df['Bone_Dice'].std():.4f}",
            "Bone_HD95_px": f"{p_df['Bone_HD95_px'].mean():.2f} ± {p_df['Bone_HD95_px'].std():.2f}"
        })
    df_patient = pd.DataFrame(patient_agg)
    patient_csv_path = os.path.join(exp_dir, "metrics", "test_patient_metrics.csv")
    df_patient.to_csv(patient_csv_path, index=False)
    print(f"✅ Saved patient metrics to: {patient_csv_path}")
    
    # 3. Overall Aggregate Test Summary
    summary = {
        "Metric": [
            "Normalized MAE",
            "Normalized PSNR (dB)",
            "Standard SSIM",
            "Masked SSIM",
            "Global HU-MAE (HU)",
            "Masked HU-MAE (HU)",
            "Bone Dice (>300 HU)",
            "Bone HD95 (px)"
        ],
        "Mean": [
            float(df_slices["MAE_norm"].mean()),
            float(df_slices["PSNR_dB"].mean()),
            float(df_slices["SSIM"].mean()),
            float(df_slices["Masked_SSIM"].mean()),
            float(df_slices["HU_MAE"].mean()),
            float(df_slices["Masked_HU_MAE"].mean()),
            float(df_slices["Bone_Dice"].mean()),
            float(df_slices["Bone_HD95_px"].mean())
        ],
        "Std": [
            float(df_slices["MAE_norm"].std()),
            float(df_slices["PSNR_dB"].std()),
            float(df_slices["SSIM"].std()),
            float(df_slices["Masked_SSIM"].std()),
            float(df_slices["HU_MAE"].std()),
            float(df_slices["Masked_HU_MAE"].std()),
            float(df_slices["Bone_Dice"].std()),
            float(df_slices["Bone_HD95_px"].std())
        ]
    }
    df_summary = pd.DataFrame(summary)
    summary_csv_path = os.path.join(exp_dir, "metrics", "test_summary_metrics.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"✅ Saved summary metrics to: {summary_csv_path}")
    
    # 4. Save Test Configuration JSON
    test_cfg = {
        "model": "Supervised ResUNet Baseline",
        "checkpoint": ckpt_path,
        "test_patients": test_patients,
        "test_slice_count": len(df_slices),
        "device": str(device),
        "metrics_summary": {
            row["Metric"]: f"{row['Mean']:.4f} ± {row['Std']:.4f}" for _, row in df_summary.iterrows()
        }
    }
    cfg_path = os.path.join(exp_dir, "configs", "test_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(test_cfg, f, indent=2)
    print(f"✅ Saved test config to: {cfg_path}")
    
    # 5. Publication Qualitative Figure
    # Selection rule: Median-error slice for each patient (strictly non-biased)
    os.makedirs(os.path.join(exp_dir, "figures"), exist_ok=True)
    fig_path = os.path.join(exp_dir, "figures", "FIG_RESUNET_QUALITATIVE_RESULTS.png")
    
    num_pts = len(test_patients)
    fig, axes = plt.subplots(num_pts, 4, figsize=(14, 3.2 * num_pts))
    if num_pts == 1:
        axes = np.expand_dims(axes, 0)
        
    for row, pid in enumerate(test_patients):
        items = patient_slices[pid]
        # Sort by HU-MAE to select median slice
        items_sorted = sorted(items, key=lambda x: x[0])
        med_idx = len(items_sorted) // 2
        med_mae, mri_img, gt_img, syn_img, sname = items_sorted[med_idx]
        
        # Display representations in [0, 1] unit space
        m_show = np.clip((mri_img + 1.0) / 2.0, 0, 1)
        g_show = np.clip((gt_img  + 1.0) / 2.0, 0, 1)
        s_show = np.clip((syn_img + 1.0) / 2.0, 0, 1)
        err_show = np.abs(s_show - g_show)
        
        # Panels: MRI | Ground Truth CT | ResUNet CT | Absolute Error
        axes[row, 0].imshow(m_show, cmap="gray")
        axes[row, 0].set_title(f"Patient {pid} (MRI)\n{sname}", fontsize=10)
        axes[row, 0].axis("off")
        
        axes[row, 1].imshow(g_show, cmap="gray")
        axes[row, 1].set_title("Ground Truth CT", fontsize=10)
        axes[row, 1].axis("off")
        
        axes[row, 2].imshow(s_show, cmap="gray")
        axes[row, 2].set_title(f"ResUNet Baseline CT\nHU-MAE: {med_mae:.1f} HU", fontsize=10, fontweight="bold", color="darkblue")
        axes[row, 2].axis("off")
        
        im_err = axes[row, 3].imshow(err_show, cmap="inferno", vmin=0, vmax=0.30)
        axes[row, 3].set_title("Absolute Error Map", fontsize=10)
        axes[row, 3].axis("off")
        
    plt.tight_layout()
    cbar = fig.colorbar(im_err, ax=axes.ravel().tolist(), shrink=0.4, pad=0.02)
    cbar.set_label("|Synthetic CT - Ground Truth|", fontsize=10)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved qualitative comparison figure to: {fig_path}")

    # 6. Plot Training Loss and Validation Curves if history exists
    tr_hist_path = os.path.join(exp_dir, "metrics", "training_history.csv")
    val_hist_path = os.path.join(exp_dir, "metrics", "validation_metrics.csv")
    if os.path.exists(tr_hist_path) and os.path.exists(val_hist_path):
        df_tr = pd.read_csv(tr_hist_path)
        df_v  = pd.read_csv(val_hist_path)
        
        # FIG_RESUNET_TRAINING_LOSS.png
        plt.figure(figsize=(8, 5))
        plt.plot(df_tr["epoch"], df_tr["train_loss"], label="Train Total Loss", color="#1f77b4", lw=2)
        plt.plot(df_v["epoch"],  df_v["val_loss"],   label="Val Total Loss",   color="#ff7f0e", lw=2)
        plt.title("ResUNet Baseline: Training and Validation Convergence", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        loss_fig_path = os.path.join(exp_dir, "figures", "FIG_RESUNET_TRAINING_LOSS.png")
        plt.savefig(loss_fig_path, dpi=300)
        plt.close()
        print(f"✅ Saved training loss figure to: {loss_fig_path}")
        
        # FIG_RESUNET_VALIDATION_METRICS.png
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        axs[0].plot(df_v["epoch"], df_v["SSIM"], color="teal", lw=2)
        axs[0].set_title("Validation SSIM (↑)", fontsize=11, fontweight="bold")
        axs[0].set_xlabel("Epoch"); axs[0].grid(True, ls="--", alpha=0.5)
        
        axs[1].plot(df_v["epoch"], df_v["PSNR_dB"], color="darkgreen", lw=2)
        axs[1].set_title("Validation PSNR (dB) (↑)", fontsize=11, fontweight="bold")
        axs[1].set_xlabel("Epoch"); axs[1].grid(True, ls="--", alpha=0.5)
        
        axs[2].plot(df_v["epoch"], df_v["Masked_HU_MAE"], color="crimson", lw=2)
        axs[2].set_title("Validation Masked HU-MAE (↓)", fontsize=11, fontweight="bold")
        axs[2].set_xlabel("Epoch"); axs[2].grid(True, ls="--", alpha=0.5)
        
        plt.tight_layout()
        metrics_fig_path = os.path.join(exp_dir, "figures", "FIG_RESUNET_VALIDATION_METRICS.png")
        plt.savefig(metrics_fig_path, dpi=300)
        plt.close()
        print(f"✅ Saved validation metrics figure to: {metrics_fig_path}")

    print("=" * 80)
    print("TEST EVALUATION COMPLETED SUCCESSFULLY.")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Frozen ResUNet on SynthRAD2023 Held-Out Test Set")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split_json", type=str, default="synthrad_experiments/baselines/resunet/configs/patient_split_seed42.json")
    parser.add_argument("--exp_dir", type=str, default="synthrad_experiments/baselines/resunet")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--base_filters", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    
    args = parser.parse_args()
    run_test_evaluation(args)
