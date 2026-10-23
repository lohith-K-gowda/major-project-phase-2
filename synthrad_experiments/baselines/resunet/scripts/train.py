# -*- coding: utf-8 -*-
"""
train.py - Full Supervised Training Pipeline for ResUNet Baseline
SynthRAD2023 Brain MRI -> CT Task
Strict Patient Isolation:
- Train: 48 patients
- Validation: 6 patients
- Test: STRICTLY FROZEN AND UNTOUCHED DURING TRAINING
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import ResUNet
from dataset import get_synthrad_loaders, find_dataset_root
from loss import ResUNetLoss
from metrics import evaluate_slice

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_training(args):
    seed_everything(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("=" * 80)
    print(" SYNTHRAD2023 RESUNET BASELINE TRAINING ")
    print("=" * 80)
    print(f"Device      : {device}")
    if torch.cuda.is_available() and not args.cpu:
        print(f"GPU Model   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM Total  : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Seed        : {args.seed}")
    print(f"Batch Size  : {args.batch_size}")
    print(f"Epochs      : {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"SSIM Weight : {args.lambda_ssim}")
    
    # 1. Output directories
    exp_dir = args.exp_dir
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "configs"), exist_ok=True)
    
    # 2. Locate dataset
    data_root = args.data_dir
    if not data_root or not os.path.exists(data_root):
        data_root = find_dataset_root()
        
    if not data_root or not os.path.exists(data_root):
        raise FileNotFoundError(
            f"Dataset not found at '{args.data_dir}'. "
            "Please specify valid --data_dir with 'mri' and 'ct' subdirectories."
        )
        
    print(f"Dataset Root: {data_root}")
    
    # 3. Load DataLoaders
    split_path = args.split_json
    train_loader, val_loader, test_loader, split_info = get_synthrad_loaders(
        base_dir=data_root,
        split_json_path=split_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers
    )
    
    print("-" * 80)
    print(f"Train Set : {len(split_info['train_patients'])} patients | {split_info['train_slices']} slices")
    print(f"Val Set   : {len(split_info['val_patients'])} patients | {split_info['val_slices']} slices")
    print(f"Test Set  : {len(split_info['test_patients'])} patients | {split_info['test_slices']} slices [FROZEN]")
    print("-" * 80)
    
    # Save training configuration
    train_config = {
        "experiment": "ResUNet Supervised Baseline",
        "dataset_path": data_root,
        "split_config": split_info,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "lambda_ssim": args.lambda_ssim,
        "seed": args.seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and not args.cpu else "CPU"
    }
    with open(os.path.join(exp_dir, "configs", "config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)
        
    # 4. Model, Loss, Optimizer
    model = ResUNet(in_channels=1, out_channels=1, base_filters=args.base_filters).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResUNet Parameters: {total_params:,}")
    
    criterion = ResUNetLoss(lambda_ssim=args.lambda_ssim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_val_loss = float("inf")
    best_epoch = 0
    
    training_history = []
    validation_history = []
    
    start_time = time.time()
    
    # 5. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_running_loss = 0.0
        train_running_l1   = 0.0
        train_running_ssim = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{args.epochs}] (Train)")
        for batch in pbar:
            mri  = batch["mri"].to(device)
            ct   = batch["ct"].to(device)
            mask = batch["mask"].to(device)
            
            optimizer.zero_grad()
            pred = model(mri)
            
            loss, logs = criterion(pred, ct, mask=mask)
            loss.backward()
            
            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_running_loss += logs["total_loss"]
            train_running_l1   += logs["l1_loss"]
            train_running_ssim += logs["ssim_loss"]
            
            pbar.set_postfix({"Loss": f"{logs['total_loss']:.4f}", "L1": f"{logs['l1_loss']:.4f}"})
            
        lr_scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        
        epoch_tr_loss = train_running_loss / len(train_loader)
        epoch_tr_l1   = train_running_l1 / len(train_loader)
        epoch_tr_ssim = train_running_ssim / len(train_loader)
        
        # 6. Validation Loop (Evaluates strictly on Validation patients)
        model.eval()
        val_running_loss = 0.0
        val_maes, val_psnrs, val_ssims, val_hu_maes, val_bone_dices = [], [], [], [], []
        
        with torch.no_grad():
            for batch in val_loader:
                mri  = batch["mri"].to(device)
                ct   = batch["ct"].to(device)
                mask = batch["mask"].to(device)
                
                pred = model(mri)
                loss, logs = criterion(pred, ct, mask=mask)
                val_running_loss += logs["total_loss"]
                
                # Calculate slice-level clinical metrics
                p_np = pred.squeeze(1).cpu().numpy()
                g_np = ct.squeeze(1).cpu().numpy()
                m_np = mask.squeeze(1).cpu().numpy()
                
                for i in range(mri.size(0)):
                    res = evaluate_slice(p_np[i], g_np[i], m_np[i])
                    val_maes.append(res["MAE_norm"])
                    val_psnrs.append(res["PSNR_dB"])
                    val_ssims.append(res["SSIM"])
                    val_hu_maes.append(res["Masked_HU_MAE"])
                    val_bone_dices.append(res["Bone_Dice"])
                    
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_mae  = np.mean(val_maes)
        epoch_val_psnr = np.mean(val_psnrs)
        epoch_val_ssim = np.mean(val_ssims)
        epoch_val_hu   = np.mean(val_hu_maes)
        epoch_val_dice = np.mean(val_bone_dices)
        
        # Save histories
        tr_rec = {
            "epoch": epoch,
            "lr": cur_lr,
            "train_loss": epoch_tr_loss,
            "train_l1": epoch_tr_l1,
            "train_ssim_loss": epoch_tr_ssim
        }
        val_rec = {
            "epoch": epoch,
            "val_loss": epoch_val_loss,
            "MAE_norm": epoch_val_mae,
            "PSNR_dB": epoch_val_psnr,
            "SSIM": epoch_val_ssim,
            "Masked_HU_MAE": epoch_val_hu,
            "Bone_Dice": epoch_val_dice
        }
        training_history.append(tr_rec)
        validation_history.append(val_rec)
        
        # Checkpoint handling
        saved_flag = ""
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_ckpt_path = os.path.join(exp_dir, "checkpoints", "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": epoch_val_loss,
                "metrics": val_rec,
                "config": train_config
            }, best_ckpt_path)
            saved_flag = " ⭐ [BEST CHECKPOINT SAVED]"
            
        print(f"Epoch [{epoch:02d}/{args.epochs}] (LR: {cur_lr:.2e}) | "
              f"Train Loss: {epoch_tr_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} (SSIM: {epoch_val_ssim:.4f}, PSNR: {epoch_val_psnr:.2f} dB, HU-MAE: {epoch_val_hu:.1f} HU){saved_flag}")

    # 7. Final Checkpoint and Export
    total_time = (time.time() - start_time) / 60.0
    print("=" * 80)
    print(f"Training completed in {total_time:.2f} minutes.")
    print(f"Best Validation Epoch: {best_epoch} (Loss: {best_val_loss:.4f})")
    
    final_ckpt_path = os.path.join(exp_dir, "checkpoints", "final_model.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "config": train_config
    }, final_ckpt_path)
    
    # Save CSV tables
    df_train = pd.DataFrame(training_history)
    df_val   = pd.DataFrame(validation_history)
    df_train.to_csv(os.path.join(exp_dir, "metrics", "training_history.csv"), index=False)
    df_val.to_csv(os.path.join(exp_dir, "metrics", "validation_metrics.csv"), index=False)
    
    print(f"Saved: {best_ckpt_path}")
    print(f"Saved: {final_ckpt_path}")
    print(f"Saved: {os.path.join(exp_dir, 'metrics', 'training_history.csv')}")
    print(f"Saved: {os.path.join(exp_dir, 'metrics', 'validation_metrics.csv')}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResUNet Baseline on SynthRAD2023")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to synthrad_png dataset")
    parser.add_argument("--split_json", type=str, default="synthrad_experiments/baselines/resunet/configs/patient_split_seed42.json")
    parser.add_argument("--exp_dir", type=str, default="synthrad_experiments/baselines/resunet")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--base_filters", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_ssim", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    
    args = parser.parse_args()
    run_training(args)
