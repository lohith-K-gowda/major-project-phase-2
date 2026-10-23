# -*- coding: utf-8 -*-
"""
build_unified_benchmark_notebook.py
Generates the publication-grade Jupyter Notebook:
SynthRAD2023_Five_Model_Unified_Benchmark.ipynb
"""

import json
import os

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()]
    })

# ==============================================================================
# CELL 0: MARKDOWN TITLE & STUDY OVERVIEW
# ==============================================================================
add_md(r'''# 🧠 SynthRAD2023: Unified Five-Model Comparative Benchmark
## Deep Learning-Based MRI-TO-CT Image Translation Using Generative Models
### Comprehensive Evaluation: ResUNet, Pix2Pix, cDDPM, SOTA-CDDPM & CCR-Diffusion
---
This notebook executes a strict, common-cohort evaluation across all **5 deep learning models** for Brain MRI-to-sCT synthesis:
1. **Model 1: ResUNet Baseline** (L1 + MSE compound regression)
2. **Model 2: Pix2Pix Generator** (Conditional GAN with PatchGAN discriminator)
3. **Model 3: cDDPM From Scratch** (Direct conditional denoising diffusion with 50 DDIM steps)
4. **Model 4: SOTA-CDDPM** (Two-stage decoupled coarse-to-fine residual diffusion with 4-fold cross-fitting)
5. **Model 5: Proposed CCR-Diffusion** (Feature-modulated residual diffusion with FiLM blocks & tissue-gated fusion)

### 📊 Metric Evaluation Protocol (as defined in `ccr-final.pdf`):
- **Slice Preprocessing:** PNG to $[-1.0, 1.0]$, Unit domain $[0.0, 1.0]$ via $(x+1)/2$, MRI body mask segmentation, Linear HU mapping $[-1000, 1500]\text{ HU}$.
- **Metrics Computed:** Masked SSIM, Global SSIM, Masked PSNR (dB), Global PSNR (dB), Masked MAE, Global MAE, Masked RMSE, Global RMSE, NCC (Masked & Global), Bone Dice (at $\ge 300\text{ HU}$, $\tau_{\text{norm}} = 0.0400$), HU-MAE, and Composite Optimization Score ($\text{MAE} + 0.25 \times (1 - \text{SSIM})$).
- **Computational & Hardware Profiling:** Inference Latency (ms/slice), Throughput (FPS), Peak GPU VRAM allocated (MB), Parameter Count (Total & Trainable in Millions), Checkpoint Size on Disk (MB).
- **Cohort Aggregation:** Slices $\to$ Patient-level means across the 6 frozen test patients (`1BA054`, `1BA097`, `1BA172`, `1BA234`, `1BA256`, `1BA307`) $\to$ Mean $\pm$ Standard Deviation.
''')

# ==============================================================================
# CELL 1: SETUP, SEEDING & PATH RESOLUTION
# ==============================================================================
add_code(r'''# =====================================================================
# STEP 1: ENVIRONMENT SETUP, REPRODUCIBILITY & AUTOMATIC PATH RESOLUTION
# =====================================================================

import os
import sys
import glob
import math
import time
import json
import shutil
import random
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from IPython.display import display, Markdown

# 1. Deterministic Seeding
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(" SYNTHRAD2023: UNIFIED FIVE-MODEL COMPARATIVE BENCHMARK ")
print("=" * 80)
print(f"PyTorch Version  : {torch.__version__}")
print(f"Device Selected  : {device}")
if torch.cuda.is_available():
    print(f"GPU Model        : {torch.cuda.get_device_name(0)}")
    vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Total VRAM       : {vram_total_gb:.2f} GB")

# 2. Output Directories
OUTPUT_DIR = Path("/kaggle/working/five_model_benchmark_results")
DIR_METRICS = OUTPUT_DIR / "metrics"
DIR_FIGURES = OUTPUT_DIR / "figures"
DIR_METRICS.mkdir(parents=True, exist_ok=True)
DIR_FIGURES.mkdir(parents=True, exist_ok=True)

# 3. Robust Path Resolver for Dataset and Checkpoints
def find_existing_path(candidate_paths, description="Target"):
    for p in candidate_paths:
        p_obj = Path(p)
        if p_obj.exists():
            print(f"[FOUND] {description}: {p_obj}")
            return p_obj
    # Recursive glob fallback in /kaggle/input
    print(f"[SEARCHING] Searching recursively for {description}...")
    for pattern in candidate_paths:
        base_name = Path(pattern).name
        matches = list(Path("/kaggle/input").rglob(base_name))
        if matches:
            print(f"[FOUND VIA GLOB] {description}: {matches[0]}")
            return matches[0]
    print(f"[WARNING] Could not resolve path for: {description}")
    return Path(candidate_paths[0])

# Dataset Root
DATASET_ROOT = find_existing_path([
    "/kaggle/input/datasets/lohithkgowda/synthrad2023-brain-2d-png/synthrad_png",
    "/kaggle/input/synthrad2023-brain-2d-png/synthrad_png",
    "/kaggle/input/synthrad_png",
    "./synthrad_png"
], "SynthRAD2023 PNG Dataset Root")

MRI_DIR = DATASET_ROOT / "mri"
CT_DIR = DATASET_ROOT / "ct"
assert MRI_DIR.exists() and CT_DIR.exists(), f"MRI or CT directory missing in {DATASET_ROOT}"

# Fixed 6 Test Patients (Held-Out Frozen Test Cohort)
TEST_PATIENTS = sorted(["1BA054", "1BA097", "1BA172", "1BA234", "1BA256", "1BA307"])
IMAGE_SIZE = 128
TIMESTEPS = 1000
DDIM_STEPS = 50

print(f"Dataset root verified. Test Patients ({len(TEST_PATIENTS)}): {TEST_PATIENTS}")
''')

# ==============================================================================
# CELL 2: PREPROCESSING, DATASET CLASS & INTENSITY MAPPINGS
# ==============================================================================
add_code(r'''# =====================================================================
# STEP 2: PREPROCESSING PIPELINE & PATIENT-ALIGNED TEST LOADER
# =====================================================================

HU_MIN = -1000.0
HU_MAX = 1500.0
HU_SPAN = HU_MAX - HU_MIN
BONE_HU_THRESHOLD = 300.0

# Normalized bone threshold in [-1.0, 1.0]:
BONE_THRESHOLD_NORM = float(2.0 * (BONE_HU_THRESHOLD - HU_MIN) / HU_SPAN - 1.0)
print(f"Intensity Range   : [{HU_MIN:.1f}, {HU_MAX:.1f}] HU (Span = {HU_SPAN:.1f} HU)")
print(f"Bone Threshold    : {BONE_HU_THRESHOLD:.1f} HU -> Normalized Threshold: {BONE_THRESHOLD_NORM:.4f}")

def extract_brain_body_mask(mri_gray):
    """
    Extracts binary body mask from MRI slice using Otsu thresholding + morphological closing.
    Clamps background to eliminate non-anatomical noise.
    """
    blur = cv2.GaussianBlur(mri_gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return (mask > 0).astype(np.float32)

class SynthRADTestDataset(Dataset):
    """
    Loads all slices belonging to the 6 fixed test patients.
    Returns MRI [-1, 1], Target CT [-1, 1], Binary Mask [0, 1], Patient ID, Slice Name.
    """
    def __init__(self, mri_dir, ct_dir, test_patients, image_size=128):
        self.image_size = image_size
        self.samples = []
        
        mri_files = sorted([f for f in os.listdir(mri_dir) if f.lower().endswith(".png")])
        for fname in mri_files:
            pid = fname.split("_")[0]
            if pid in test_patients:
                ct_path = ct_dir / fname
                if ct_path.exists():
                    self.samples.append({
                        "pid": pid,
                        "fname": fname,
                        "mri_path": mri_dir / fname,
                        "ct_path": ct_path
                    })
        print(f"[TEST COHORT] Loaded {len(self.samples)} slices across {len(test_patients)} test patients.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        mri_img = cv2.imread(str(item["mri_path"]), cv2.IMREAD_GRAYSCALE)
        ct_img = cv2.imread(str(item["ct_path"]), cv2.IMREAD_GRAYSCALE)
        
        mask = extract_brain_body_mask(mri_img)
        mri_masked = mri_img * mask
        
        sz = (self.image_size, self.image_size)
        mri_res = cv2.resize(mri_masked, sz, interpolation=cv2.INTER_LINEAR)
        ct_res = cv2.resize(ct_img, sz, interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(mask, sz, interpolation=cv2.INTER_NEAREST)
        
        # Scale [0, 255] -> [-1.0, 1.0]
        mri_norm = (mri_res / 255.0) * 2.0 - 1.0
        ct_norm = (ct_res / 255.0) * 2.0 - 1.0
        mri_norm[mask_res < 0.5] = -1.0
        
        return {
            "mri": torch.from_numpy(mri_norm).unsqueeze(0).float(),
            "ct": torch.from_numpy(ct_norm).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask_res).unsqueeze(0).float(),
            "pid": item["pid"],
            "fname": item["fname"]
        }

test_dataset = SynthRADTestDataset(MRI_DIR, CT_DIR, TEST_PATIENTS, image_size=IMAGE_SIZE)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
''')

# ==============================================================================
# CELL 3: METRIC FUNCTIONS AS DEFINED IN CCR-FINAL.PDF
# ==============================================================================
add_code(r'''# =====================================================================
# STEP 3: CLINICAL EVALUATION METRICS (EXACT IMPLEMENTATION FROM CCR-FINAL)
# =====================================================================

def calculate_ncc(a, b, valid_mask=None):
    """Zero-mean Normalized Cross-Correlation (Pearson correlation)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        a, b = a[valid], b[valid]
    else:
        a, b = a.ravel(), b.ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 2: return np.nan
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return float(np.sum(a * b) / denom) if denom >= 1e-12 else np.nan

def calculate_bone_dice(pred_norm, gt_norm, body_mask, threshold_norm=BONE_THRESHOLD_NORM):
    """
    Computes Bone Dice coefficient inside MRI body mask at threshold >= 300 HU (0.0400 norm).
    """
    body = np.asarray(body_mask) > 0.5
    pred_bone = (np.asarray(pred_norm) > threshold_norm) & body
    gt_bone = (np.asarray(gt_norm) > threshold_norm) & body
    denom = int(pred_bone.sum() + gt_bone.sum())
    if denom == 0: return 1.0
    intersection = np.logical_and(pred_bone, gt_bone).sum()
    return float(2.0 * intersection / denom)

def compute_all_slice_metrics(pred_norm, gt_norm, body_mask):
    """
    Computes complete suite of image quality & radiotherapy metrics for a single slice.
    """
    pred = np.clip(np.asarray(pred_norm, dtype=np.float32), -1.0, 1.0)
    gt = np.clip(np.asarray(gt_norm, dtype=np.float32), -1.0, 1.0)
    body = np.asarray(body_mask) > 0.5
    if not body.any():
        body = np.ones_like(pred, dtype=bool)
        
    # Unit domain [0, 1]
    p = (pred + 1.0) / 2.0
    g = (gt + 1.0) / 2.0
    diff = p - g
    
    # 1. Masked Metrics (Inside body)
    masked_diff = diff[body]
    masked_mae = float(np.mean(np.abs(masked_diff)))
    masked_mse = float(np.mean(masked_diff ** 2))
    masked_rmse = float(np.sqrt(masked_mse))
    _, ssim_map = compute_ssim(g, p, data_range=1.0, full=True)
    masked_ssim = float(np.mean(ssim_map[body]))
    masked_psnr = float(10.0 * np.log10(1.0 / masked_mse)) if masked_mse > 0 else float("inf")
    
    # 2. Global Metrics (Full frame)
    global_mae = float(np.mean(np.abs(diff)))
    global_mse = float(np.mean(diff ** 2))
    global_rmse = float(np.sqrt(global_mse))
    global_ssim = float(np.mean(ssim_map))
    global_psnr = float(10.0 * np.log10(1.0 / global_mse)) if global_mse > 0 else float("inf")
    
    # 3. Bone Dice & NCC
    bone_dice = calculate_bone_dice(pred, gt, body, BONE_THRESHOLD_NORM)
    ncc_masked = calculate_ncc(pred, gt, body)
    ncc_global = calculate_ncc(pred, gt)
    
    # 4. HU-MAE & Composite Score
    hu_mae = masked_mae * HU_SPAN
    composite_score = masked_mae + 0.25 * (1.0 - masked_ssim)
    
    return {
        "Masked_MAE_norm": masked_mae,
        "Masked_RMSE_norm": masked_rmse,
        "Masked_SSIM": masked_ssim,
        "Masked_PSNR_dB": masked_psnr,
        "Global_MAE_norm": global_mae,
        "Global_RMSE_norm": global_rmse,
        "Global_SSIM": global_ssim,
        "Global_PSNR_dB": global_psnr,
        "NCC_masked": ncc_masked,
        "NCC_global": ncc_global,
        "Bone_Dice_300HU": bone_dice,
        "HU_MAE": hu_mae,
        "Composite_Score": composite_score
    }
''')

# ==============================================================================
# CELL 4: ARCHITECTURAL DEFINITIONS FOR ALL 5 MODELS
# ==============================================================================
add_code(r'''# =====================================================================
# STEP 4: MODEL ARCHITECTURES (EXACT STRUCTURAL REPRODUCIBILITY)
# =====================================================================

# ----------------- 1. Sinusoidal Position Embeddings -----------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

# ----------------- 2. MODEL 1: ResUNet Baseline (4-Stage, GroupNorm, 64 Base Filters) -----------------
class MSEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
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

class ResUNetBaseline(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super().__init__()
        f = base_filters
        self.enc1 = MSEResBlock(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = MSEResBlock(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = MSEResBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = MSEResBlock(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = nn.Sequential(
            MSEResBlock(f * 8, f * 8),
            MSEResBlock(f * 8, f * 8)
        )
        
        self.up4 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec4 = MSEResBlock(f * 8 + f * 4, f * 4)
        
        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec3 = MSEResBlock(f * 4 + f * 2, f * 2)
        
        self.up2 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec2 = MSEResBlock(f * 2 + f, f)
        
        self.up1 = nn.ConvTranspose2d(f, f, kernel_size=2, stride=2)
        self.dec1 = MSEResBlock(f + f, f)
        
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)
        self.final_act = nn.Tanh()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        b = self.bottleneck(self.pool4(e4))
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.final_act(self.final_conv(d1))


# ----------------- 3. MODEL 2: Pix2Pix Generator -----------------
class Pix2PixDown(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if norm: layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class Pix2PixUp(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        if dropout > 0.0: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip): return torch.cat([self.model(x), skip], dim=1)

class Pix2PixGenerator(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.d1 = Pix2PixDown(in_ch, 64, norm=False)
        self.d2 = Pix2PixDown(64, 128)
        self.d3 = Pix2PixDown(128, 256)
        self.d4 = Pix2PixDown(256, 512)
        self.d5 = Pix2PixDown(512, 512)
        self.d6 = Pix2PixDown(512, 512)
        self.d7 = Pix2PixDown(512, 512)
        self.u1 = Pix2PixUp(512, 512, dropout=0.5)
        self.u2 = Pix2PixUp(1024, 512, dropout=0.5)
        self.u3 = Pix2PixUp(1024, 512)
        self.u4 = Pix2PixUp(1024, 256)
        self.u5 = Pix2PixUp(512, 128)
        self.u6 = Pix2PixUp(256, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, out_ch, 4, 2, 1), nn.Tanh())
    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6)
        u1 = self.u1(d7, d6); u2 = self.u2(u1, d5); u3 = self.u3(u2, d4)
        u4 = self.u4(u3, d3); u5 = self.u5(u4, d2); u6 = self.u6(u5, d1)
        return self.final(u6)

# ----------------- 4. MODEL 3: cDDPM From Scratch -----------------
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(8 if out_ch % 8 == 0 else 4, out_ch)
        self.act1  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(8 if out_ch % 8 == 0 else 4, out_ch)
        self.act2  = nn.SiLU()
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act1(self.gn1(self.conv1(x)))
        h = h + self.act1(self.time_mlp(t_emb))[:, :, None, None]
        h = self.act2(self.gn2(self.conv2(h)))
        return h + self.shortcut(x)

class StandardConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32, time_emb_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        c = base_channels
        self.enc1  = ResidualConvBlock(in_channels, c, time_emb_dim)
        self.down1 = nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1)
        self.enc2  = ResidualConvBlock(c, c * 2, time_emb_dim)
        self.down2 = nn.Conv2d(c * 2, c * 2, kernel_size=3, stride=2, padding=1)
        self.enc3  = ResidualConvBlock(c * 2, c * 4, time_emb_dim)
        self.down3 = nn.Conv2d(c * 4, c * 4, kernel_size=3, stride=2, padding=1)

        self.bot1  = ResidualConvBlock(c * 4, c * 8, time_emb_dim)
        self.bot2  = ResidualConvBlock(c * 8, c * 8, time_emb_dim)

        self.up3   = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3  = ResidualConvBlock(c * 4 + c * 4, c * 4, time_emb_dim)
        self.up2   = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2  = ResidualConvBlock(c * 2 + c * 2, c * 2, time_emb_dim)
        self.up1   = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1  = ResidualConvBlock(c + c, c, time_emb_dim)
        self.final_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x, t_emb); d1 = self.down1(e1)
        e2 = self.enc2(d1, t_emb); d2 = self.down2(e2)
        e3 = self.enc3(d2, t_emb); d3 = self.down3(e3)
        b = self.bot2(self.bot1(d3, t_emb), t_emb)
        d3_out = self.dec3(torch.cat([self.up3(b), e3], dim=1), t_emb)
        d2_out = self.dec2(torch.cat([self.up2(d3_out), e2], dim=1), t_emb)
        d1_out = self.dec1(torch.cat([self.up1(d2_out), e1], dim=1), t_emb)
        return self.final_conv(d1_out)


# ----------------- 5. MODEL 4: SOTA cDDPM (ResAttentionUNet) -----------------
class AttentionGate(nn.Module):
    """Suppresses background noise and highlights skull/brain boundaries."""
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.GroupNorm(8, f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.GroupNorm(8, f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.w_g(g) + self.w_x(x)))

class RegularizedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.10):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout2d(p=dropout)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.gn1(self.conv1(x)))
        h = h + self.act(self.time_mlp(t_emb))[:, :, None, None]
        h = self.dropout(self.act(self.gn2(self.conv2(h))))
        return h + self.shortcut(x)

class ResAttentionUNet(nn.Module):
    """Attention-Gated Residual U-Net accepting 3 channels [x_t, mri, mask]."""
    def __init__(self, in_channels=3, out_channels=1, time_emb_dim=64, dropout=0.0):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        self.enc1 = RegularizedResBlock(in_channels, 64, time_emb_dim, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = RegularizedResBlock(64, 128, time_emb_dim, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = RegularizedResBlock(128, 256, time_emb_dim, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.bot1 = RegularizedResBlock(256, 512, time_emb_dim, dropout)
        self.bot2 = RegularizedResBlock(512, 256, time_emb_dim, dropout)
        self.gate3 = AttentionGate(256, 256, 128)
        self.gate2 = AttentionGate(128, 128, 64)
        self.gate1 = AttentionGate(64,  64,  32)
        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = RegularizedResBlock(256 + 256, 128, time_emb_dim, dropout)
        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = RegularizedResBlock(128 + 128, 64, time_emb_dim, dropout)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = RegularizedResBlock(64 + 64, 64, time_emb_dim, dropout)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool1(e1), t_emb)
        e3 = self.enc3(self.pool2(e2), t_emb)
        b  = self.bot2(self.bot1(self.pool3(e3), t_emb), t_emb)
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, self.gate3(u3, e3)], dim=1), t_emb)
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, self.gate2(u2, e2)], dim=1), t_emb)
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, self.gate1(u1, e1)], dim=1), t_emb)
        return self.final(d1)


# ----------------- 6. MODEL 5: CCR-Diffusion (FiLM-Modulated) -----------------
class Stage1ResUNet(nn.Module):
    """
    Stage-1 Anatomical Anchor ResUNet:
    - Supervised regression mapping MRI to structural Coarse CT in [-1, 1].
    - Contains extract_encoder_features() to feed multi-scale conditioning
      feature maps directly into Stage-2 FiLM modulation blocks.
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, in_ch=None, out_ch=None):
        super().__init__()
        in_channels = in_ch if in_ch is not None else in_channels
        out_channels = out_ch if out_ch is not None else out_channels
        f = base_filters
        self.enc1  = MSEResBlock(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2  = MSEResBlock(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3  = MSEResBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4  = MSEResBlock(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            MSEResBlock(f * 8, f * 8),
            MSEResBlock(f * 8, f * 8)
        )

        self.up4  = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec4 = MSEResBlock(f * 8 + f * 4, f * 4)
        self.up3  = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec3 = MSEResBlock(f * 4 + f * 2, f * 2)
        self.up2  = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec2 = MSEResBlock(f * 2 + f, f)
        self.up1  = nn.ConvTranspose2d(f, f, kernel_size=2, stride=2)
        self.dec1 = MSEResBlock(f + f, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)
        self.final_act  = nn.Tanh()

    def extract_encoder_features(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        return {
            "enc1": e1,
            "enc2": e2,
            "enc3": e3,
            "enc4": e4,
            "bot":  b
        }

    def forward(self, x):
        feats = self.extract_encoder_features(x)
        e1, e2, e3, e4, b = feats["enc1"], feats["enc2"], feats["enc3"], feats["enc4"], feats["bot"]
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final_act(self.final_conv(d1))

class FiLMBlock(nn.Module):
    def __init__(self, feature_channels, cond_channels):
        super().__init__()
        self.mod_proj = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(cond_channels, feature_channels * 2, kernel_size=1)
        )
        nn.init.zeros_(self.mod_proj[-1].weight)
        nn.init.zeros_(self.mod_proj[-1].bias)

    def forward(self, x, cond_feat):
        if cond_feat.shape[2:] != x.shape[2:]:
            cond_feat = F.interpolate(cond_feat, size=x.shape[2:], mode="bilinear", align_corners=False)
        mod = self.mod_proj(cond_feat)
        gamma, beta = torch.chunk(mod, 2, dim=1)
        return (1.0 + gamma) * x + beta

class DiffusionResBlockFiLM(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_channels=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(8 if out_ch % 8 == 0 else 4, out_ch)
        self.act1  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(8 if out_ch % 8 == 0 else 4, out_ch)
        self.act2  = nn.SiLU()
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.film = FiLMBlock(out_ch, cond_channels) if cond_channels is not None else None

    def forward(self, x, t_emb, cond_feat=None):
        h = self.act1(self.gn1(self.conv1(x)))
        h = h + self.act1(self.time_mlp(t_emb))[:, :, None, None]
        h = self.act2(self.gn2(self.conv2(h)))
        out = h + self.shortcut(x)
        if self.film is not None and cond_feat is not None:
            out = self.film(out, cond_feat)
        return out

class ConditionalDiffusionUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=32, time_emb_dim=64,
                 cond_channels=(32, 64, 128, 256), in_ch=None, out_ch=None, base_ch=None):
        super().__init__()
        in_channels = in_ch if in_ch is not None else in_channels
        out_channels = out_ch if out_ch is not None else out_channels
        base_channels = base_ch if base_ch is not None else base_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        c1, c2, c3, c4 = cond_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        c = base_channels

        self.enc1  = DiffusionResBlockFiLM(in_channels, c, time_emb_dim)
        self.down1 = nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1)
        self.enc2  = DiffusionResBlockFiLM(c, c * 2, time_emb_dim)
        self.down2 = nn.Conv2d(c * 2, c * 2, kernel_size=3, stride=2, padding=1)
        self.enc3  = DiffusionResBlockFiLM(c * 2, c * 4, time_emb_dim)
        self.down3 = nn.Conv2d(c * 4, c * 4, kernel_size=3, stride=2, padding=1)
        self.enc4  = DiffusionResBlockFiLM(c * 4, c * 8, time_emb_dim)
        self.down4 = nn.Conv2d(c * 8, c * 8, kernel_size=3, stride=2, padding=1)

        self.bot1  = DiffusionResBlockFiLM(c * 8, c * 8, time_emb_dim)
        self.bot2  = DiffusionResBlockFiLM(c * 8, c * 8, time_emb_dim)

        self.up4   = nn.ConvTranspose2d(c * 8, c * 8, kernel_size=2, stride=2)
        self.dec4  = DiffusionResBlockFiLM(c * 8 + c * 8, c * 4, time_emb_dim, cond_channels=c4)
        self.up3   = nn.ConvTranspose2d(c * 4, c * 4, kernel_size=2, stride=2)
        self.dec3  = DiffusionResBlockFiLM(c * 4 + c * 4, c * 2, time_emb_dim, cond_channels=c3)
        self.up2   = nn.ConvTranspose2d(c * 2, c * 2, kernel_size=2, stride=2)
        self.dec2  = DiffusionResBlockFiLM(c * 2 + c * 2, c, time_emb_dim, cond_channels=c2)
        self.up1   = nn.ConvTranspose2d(c, c, kernel_size=2, stride=2)
        self.dec1  = DiffusionResBlockFiLM(c + c, c, time_emb_dim, cond_channels=c1)

        self.final_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, z_t, t, cond_features):
        t_emb = self.time_mlp(t)
        e1 = self.enc1(z_t, t_emb); d1 = self.down1(e1)
        e2 = self.enc2(d1, t_emb); d2 = self.down2(e2)
        e3 = self.enc3(d2, t_emb); d3 = self.down3(e3)
        e4 = self.enc4(d3, t_emb); d4 = self.down4(e4)
        b = self.bot2(self.bot1(d4, t_emb), t_emb)
        u4 = self.up4(b)
        d4_out = self.dec4(torch.cat([u4, e4], dim=1), t_emb, cond_feat=cond_features["enc4"])
        u3 = self.up3(d4_out)
        d3_out = self.dec3(torch.cat([u3, e3], dim=1), t_emb, cond_feat=cond_features["enc3"])
        u2 = self.up2(d3_out)
        d2_out = self.dec2(torch.cat([u2, e2], dim=1), t_emb, cond_feat=cond_features["enc2"])
        u1 = self.up1(d2_out)
        d1_out = self.dec1(torch.cat([u1, e1], dim=1), t_emb, cond_feat=cond_features["enc1"])
        return self.final_conv(d1_out)
''')

# ==============================================================================
# CELL 5: DDIM REVERSE SAMPLERS (MODELS 3, 4, 5)
# ==============================================================================
add_code(r'''# =====================================================================
# STEP 5: DDIM REVERSE DIFFUSION SAMPLERS (DETERMINISTIC 50 STEPS)
# =====================================================================

# Standard Linear Beta Schedule
betas = torch.linspace(1e-4, 0.02, TIMESTEPS, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

@torch.inference_mode()
def sample_cddpm_scratch_ddim(model, mri, mask, steps=50):
    """Sampler for Model 3 (cDDPM From Scratch conditioned on [x_t, MRI, Mask])."""
    model.eval()
    b, _, h, w = mri.shape
    x_t = torch.randn((b, 1, h, w), device=device)
    timesteps = sorted(torch.linspace(0, TIMESTEPS - 1, steps).long().tolist(), reverse=True)
    
    for i, t_val in enumerate(timesteps):
        t = torch.full((b,), t_val, device=device, dtype=torch.long)
        pred_noise = model(torch.cat([x_t, mri, mask], dim=1), t)
        
        a_cur = alpha_bars[t_val]
        a_prev = alpha_bars[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
        
        pred_x0 = torch.clamp((x_t - torch.sqrt(1.0 - a_cur) * pred_noise) / (torch.sqrt(a_cur) + 1e-8), -1.0, 1.0)
        pred_noise_adj = (x_t - torch.sqrt(a_cur) * pred_x0) / (torch.sqrt(1.0 - a_cur) + 1e-8)
        x_t = torch.sqrt(a_prev) * pred_x0 + torch.clamp(1.0 - a_prev, min=0.0).sqrt() * pred_noise_adj
    return torch.clamp(x_t, -1.0, 1.0)


@torch.inference_mode()
def sample_sota_ddim(model, mri, mask, steps=50):
    """DDIM 50-Step Deterministic Reverse Sampler for SOTA-CDDPM (Model 4)."""
    model.eval()
    b, _, h, w = mri.shape
    x_t = torch.randn((b, 1, h, w), device=device)
    timesteps = sorted(torch.linspace(0, TIMESTEPS - 1, steps).long().tolist(), reverse=True)
    
    for i, t_val in enumerate(timesteps):
        t = torch.full((b,), t_val, device=device, dtype=torch.long)
        pred_noise = model(torch.cat([x_t, mri, mask], dim=1), t)
        
        a_cur = alpha_bars[t_val]
        a_prev = alpha_bars[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
        
        pred_x0 = torch.clamp((x_t - torch.sqrt(1.0 - a_cur) * pred_noise) / (torch.sqrt(a_cur) + 1e-8), -1.0, 1.0)
        dir_xt = torch.clamp(1.0 - a_prev, min=0.0).sqrt() * pred_noise
        x_t = torch.sqrt(a_prev) * pred_x0 + dir_xt

    x_t = torch.where(mask > 0.5, x_t, -1.0)
    return torch.clamp(x_t, -1.0, 1.0)

@torch.inference_mode()
def sample_ccr_diffusion_ddim(diff_model, stage1_model, mri, mask, steps=50, residual_scale=0.16):
    """Deterministic DDIM reverse sampling for Model 5 (CCR-Diffusion with FiLM features)."""
    diff_model.eval(); stage1_model.eval()
    dev = mri.device
    b, _, h, w = mri.shape

    # 1. Coarse CT & multi-scale feature representations from Stage 1 Anchor
    coarse_ct = stage1_model(mri)
    cond_features = stage1_model.extract_encoder_features(mri)

    # 2. Diffusion Reverse Residual Denoising
    z_t = torch.randn((b, 1, h, w), device=dev)
    timesteps = sorted(torch.linspace(0, TIMESTEPS - 1, steps).long().tolist(), reverse=True)

    for i, t_val in enumerate(timesteps):
        t = torch.full((b,), t_val, device=dev, dtype=torch.long)
        model_in = torch.cat([z_t, mask], dim=1)
        pred_noise = diff_model(model_in, t, cond_features)

        a_cur = alpha_bars[t_val]
        a_prev = alpha_bars[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=dev)

        pred_z0 = torch.clamp((z_t - torch.sqrt(1.0 - a_cur) * pred_noise) / (torch.sqrt(a_cur) + 1e-8), -6.0, 6.0)
        pred_noise_adj = (z_t - torch.sqrt(a_cur) * pred_z0) / (torch.sqrt(1.0 - a_cur) + 1e-8)
        dir_zt = torch.clamp(1.0 - a_prev, min=0.0).sqrt() * pred_noise_adj
        z_t = torch.sqrt(a_prev) * pred_z0 + dir_zt

    # 3. Final CT reconstruction: sCT = clip(Coarse + s * z_0, -1.0, 1.0)
    pred_residual = z_t * residual_scale
    sct = torch.clamp(coarse_ct + pred_residual, -1.0, 1.0)
    sct = torch.where(mask > 0.5, sct, torch.full_like(sct, -1.0))
    return torch.clamp(sct, -1.0, 1.0)
''')

# ==============================================================================
# CELL 6: CHECKPOINT LOADING & HARDWARE PROFILER
# ==============================================================================
add_code(r'''# =====================================================================
# STEP 6: CHECKPOINT DISCOVERY, MODEL INITIALIZATION & HARDWARE PROFILING
# =====================================================================

def profile_model_hardware(model, input_tensors, forward_fn, name="Model", warmup=5, iters=20):
    """
    Measures: Total & Trainable Params, Latency per slice (ms), FPS, and Peak GPU VRAM (MB).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    
    # Warmup
    for _ in range(warmup):
        _ = forward_fn(*input_tensors)
        
    # Timing with CUDA Events
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iters):
            _ = forward_fn(*input_tensors)
        end_event.record()
        torch.cuda.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
    else:
        t0 = time.time()
        for _ in range(iters):
            _ = forward_fn(*input_tensors)
        total_time_ms = (time.time() - t0) * 1000.0
        
    latency_ms = total_time_ms / iters
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0
    
    return {
        "Model": name,
        "Total_Params_M": total_params / 1e6,
        "Trainable_Params_M": trainable_params / 1e6,
        "Latency_ms": latency_ms,
        "Throughput_FPS": fps,
        "Peak_VRAM_MB": peak_vram_mb
    }

print("Profiler initialized.")
''')

# ==============================================================================
# CELL 7: INFERENCE & EVALUATION ACROSS ALL 5 MODELS
# ==============================================================================
add_code(r'''# =====================================================================
# STEP 7: EXECUTE INFERENCE & COMPUTE METRICS FOR ALL 5 MODELS
# =====================================================================

# Resolve Checkpoint Paths (Matching user's attached Kaggle deliverables)
CKPT_RESUNET = find_existing_path([
    "/kaggle/input/datasets/sonumehaboob/final-study-model-comparisions/msel1-resunet/resunet_l1_mse_baseline/checkpoints/best_resunet_l1_mse.pth",
    "/kaggle/input/msel1-resunet/resunet_l1_mse_baseline/checkpoints/best_resunet_l1_mse.pth",
    "/kaggle/input/**/best_resunet_l1_mse.pth"
], "Model 1 (ResUNet Checkpoint)")

CKPT_PIX2PIX = find_existing_path([
    "/kaggle/input/synthrad_model2_pix2pix_deliverables/checkpoints/best_pix2pix_generator.pth",
    "/kaggle/input/**/best_pix2pix_generator.pth"
], "Model 2 (Pix2Pix Checkpoint)")

CKPT_CDDPM_SCRATCH = find_existing_path([
    "/kaggle/input/synthrad_model3_cddpm_from_scratch_deliverables/checkpoints/best_model3_cddpm_from_scratch.pth",
    "/kaggle/input/**/best_model3_cddpm_from_scratch.pth"
], "Model 3 (cDDPM Scratch Checkpoint)")

CKPT_SOTA_CDDPM = find_existing_path([
    "/kaggle/input/**/best_cddpm_synthrad_sota.pth",
    "/kaggle/input/**/synthrad_model4*/**/best*.pth"
], "Model 4 (SOTA-CDDPM Checkpoint)")

CKPT_CCR = find_existing_path([
    "/kaggle/input/synthrad_ccr_diffusion_complete_results/synthrad_ccr_diffusion/checkpoints/best_checkpoint.pth",
    "/kaggle/input/synthrad_ccr_diffusion_final_deliverables/**/best_checkpoint.pth",
    "/kaggle/input/**/best_checkpoint.pth"
], "Model 5 (CCR-Diffusion Checkpoint)")

CKPT_CCR_COARSE = find_existing_path([
    "/kaggle/input/synthrad_ccr_diffusion_complete_results/synthrad_ccr_diffusion/checkpoints/coarse_resunet/coarse_resunet_fold_1.pth",
    "/kaggle/input/**/coarse_resunet/coarse_resunet_fold_1.pth",
    "/kaggle/input/**/coarse_resunet_fold_*.pth"
], "Model 5 Coarse Anchor Checkpoint")

# Load Weights Helper
def safe_load_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else (
            ckpt["generator"] if isinstance(ckpt, dict) and "generator" in ckpt else ckpt)
    # Strip unexpected module prefixes
    cleaned = {k.replace("module.", "").replace("generator.", "").replace("net_G.", ""): v for k, v in state.items()}
    try:
        model.load_state_dict(cleaned, strict=True)
        print(f"[LOADED STRICT] {ckpt_path.name}")
    except Exception as e:
        model.load_state_dict(cleaned, strict=False)
        print(f"[LOADED RELAXED] {ckpt_path.name}")
    model.eval()
    return model

# Initialize and Load Models
m1_resunet = safe_load_weights(ResUNetBaseline(in_channels=1, out_channels=1, base_filters=64).to(device), CKPT_RESUNET)
m2_pix2pix = safe_load_weights(Pix2PixGenerator(in_ch=1, out_ch=1).to(device), CKPT_PIX2PIX)
m3_cddpm = safe_load_weights(StandardConditionalUNet(in_channels=3, out_channels=1, base_channels=32, time_emb_dim=64).to(device), CKPT_CDDPM_SCRATCH)
m4_sota = safe_load_weights(ResAttentionUNet(in_channels=3, out_channels=1, time_emb_dim=64, dropout=0.0).to(device), CKPT_SOTA_CDDPM)

m5_stage1 = safe_load_weights(Stage1ResUNet(in_channels=1, out_channels=1, base_filters=32).to(device), CKPT_CCR_COARSE)
m5_ccr = safe_load_weights(ConditionalDiffusionUNet(in_channels=2, out_channels=1, base_channels=32, time_emb_dim=64).to(device), CKPT_CCR)


# Hardware Profiling on dummy slice
dummy_mri = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
dummy_mask = torch.ones(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)

hw_results = []
hw_results.append(profile_model_hardware(m1_resunet, (dummy_mri,), lambda x: m1_resunet(x), "ResUNet Baseline"))
hw_results.append(profile_model_hardware(m2_pix2pix, (dummy_mri,), lambda x: m2_pix2pix(x), "Pix2Pix Generator"))
hw_results.append(profile_model_hardware(m3_cddpm, (dummy_mri, dummy_mask), lambda x, m: sample_cddpm_scratch_ddim(m3_cddpm, x, m, 50), "cDDPM From Scratch"))
hw_results.append(profile_model_hardware(m4_sota, (dummy_mri, dummy_mask), lambda x, m: sample_sota_ddim(m4_sota, x, m, 50), "SOTA-CDDPM (Residual)"))
hw_results.append(profile_model_hardware(m5_ccr, (dummy_mri, dummy_mask), lambda x, m: sample_ccr_diffusion_ddim(m5_ccr, m5_stage1, x, m, 50), "CCR-Diffusion (Proposed)"))

hw_df = pd.DataFrame(hw_results)
hw_df.to_csv(DIR_METRICS / "hardware_profiling_latency_vram_summary.csv", index=False)
print("\n" + "=" * 80)
print(" HARDWARE & COMPUTATIONAL PROFILING SUMMARY ")
print("=" * 80)
display(hw_df)

# Inference and Metrics Loop across all test slices
slice_records = []
all_qualitative_predictions = {}

print("\nExecuting common test-cohort inference across 363 slices...")
for sample in tqdm(test_loader, desc="Benchmarking All 5 Models"):
    mri = sample["mri"].to(device)
    gt = sample["ct"].to(device)
    mask = sample["mask"].to(device)
    pid = sample["pid"][0]
    fname = sample["fname"][0]
    
    with torch.no_grad():
        p1 = m1_resunet(mri)
        p2 = m2_pix2pix(mri)
        p3 = sample_cddpm_scratch_ddim(m3_cddpm, mri, mask, DDIM_STEPS)
        p4 = sample_sota_ddim(m4_sota, mri, mask, DDIM_STEPS)
        p5 = sample_ccr_diffusion_ddim(m5_ccr, m5_stage1, mri, mask, DDIM_STEPS)

        
    preds = {
        "ResUNet Baseline": p1[0, 0].cpu().numpy(),
        "Pix2Pix Generator": p2[0, 0].cpu().numpy(),
        "cDDPM From Scratch": p3[0, 0].cpu().numpy(),
        "SOTA-CDDPM (Residual)": p4[0, 0].cpu().numpy(),
        "CCR-Diffusion (Proposed)": p5[0, 0].cpu().numpy()
    }
    
    gt_np = gt[0, 0].cpu().numpy()
    mask_np = mask[0, 0].cpu().numpy()
    mri_np = mri[0, 0].cpu().numpy()
    
    all_qualitative_predictions[fname] = {
        "mri": mri_np, "gt": gt_np, "mask": mask_np, "pid": pid, "preds": preds
    }
    
    for model_name, pred_arr in preds.items():
        metrics = compute_all_slice_metrics(pred_arr, gt_np, mask_np)
        slice_records.append({
            "model": model_name,
            "patient_id": pid,
            "slice_name": fname,
            **metrics
        })

slice_df = pd.DataFrame(slice_records)
slice_df.to_csv(DIR_METRICS / "test_metrics_all_models_slice_level.csv", index=False)
print(f"Computed metrics for {len(slice_df)} model-slice evaluations.")
''')

# ==============================================================================
# CELL 8: PATIENT-LEVEL AGGREGATION & REPORT MATRIX
# ==============================================================================
add_code(r'''# =====================================================================
# STEP 8: PATIENT-LEVEL AGGREGATION & MASTER COMPARATIVE TABLE
# =====================================================================

# Tier 1: Patient-Level Means (Averages across slices of each patient)
metric_cols = [c for c in slice_df.columns if c not in ["model", "patient_id", "slice_name"]]
patient_df = slice_df.groupby(["model", "patient_id"], as_index=False)[metric_cols].mean()
patient_df.to_csv(DIR_METRICS / "test_metrics_all_models_patient_level.csv", index=False)

# Tier 2: Cohort Statistics (Mean and Standard Deviation across the 6 test patients)
summary_df = patient_df.groupby("model")[metric_cols].agg(["mean", "std"])
summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]
summary_df = summary_df.reset_index()
summary_df.to_csv(DIR_METRICS / "test_metrics_all_models_mean_std_summary.csv", index=False)

# Master Formatted Table for Project Report
display_metrics = [
    ("Masked_SSIM", "Masked SSIM (↑)", 4),
    ("Global_SSIM", "Global SSIM (↑)", 4),
    ("Masked_PSNR_dB", "Masked PSNR dB (↑)", 2),
    ("Global_PSNR_dB", "Global PSNR dB (↑)", 2),
    ("Masked_MAE_norm", "Masked MAE norm (↓)", 4),
    ("HU_MAE", "HU MAE (↓)", 1),
    ("Bone_Dice_300HU", "Bone Dice >= 300 HU (↑)", 4),
    ("Composite_Score", "Composite Score (↓)", 4)
]

report_rows = []
for idx, row in summary_df.iterrows():
    m_name = row["model"]
    row_data = {"Model": m_name}
    for m_key, m_label, digits in display_metrics:
        m_mean = row[f"{m_key}_mean"]
        m_std = row[f"{m_key}_std"]
        row_data[m_label] = f"{m_mean:.{digits}f} ± {m_std:.{digits}f}"
    report_rows.append(row_data)

master_report_df = pd.DataFrame(report_rows)
print("\n" + "=" * 140)
print(" MASTER 5-MODEL CLINICAL COMPARISON (PATIENT-LEVEL MEAN ± SD ON FROZEN TEST SET) ")
print("=" * 140)
display(master_report_df)
''')

# ==============================================================================
# CELL 9: VISUALIZATIONS & QUALITATIVE DELIVERABLES
# ==============================================================================
add_code(r'''# =====================================================================
# STEP 9: PUBLICATION COMPARISON PLOTS & SAME-SLICE QUALITATIVE FIGURES
# =====================================================================

# 1. Multi-Metric Bar Charts
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

plot_targets = [
    ("Masked_SSIM_mean", "Masked_SSIM_std", "Masked SSIM (Structural Fidelity)", "higher"),
    ("Masked_PSNR_dB_mean", "Masked_PSNR_dB_std", "Masked PSNR (dB)", "higher"),
    ("HU_MAE_mean", "HU_MAE_std", "Radiometric Error: HU MAE (HU)", "lower"),
    ("Bone_Dice_300HU_mean", "Bone_Dice_300HU_std", "Bone Dice (>= 300 HU)", "higher")
]

models = summary_df["model"].tolist()
colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]

for i, (m_col, s_col, title, direction) in enumerate(plot_targets):
    ax = axes[i]
    means = summary_df[m_col].values
    stds = summary_df[s_col].values
    bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    
    # Highlight Winner
    win_idx = np.argmin(means) if direction == "lower" else np.argmax(means)
    bars[win_idx].set_edgecolor("gold")
    bars[win_idx].set_linewidth(3.0)
    
    ax.set_title(f"{title} (Best: {models[win_idx]})", fontsize=12, fontweight="bold")
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
fig_path = DIR_FIGURES / "five_model_metric_comparisons.png"
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"[FIGURE SAVED] {fig_path}")

# 2. Latency vs. Masked SSIM Trade-Off Scatter Plot
merged_hw_metrics = pd.merge(summary_df, hw_df, left_on="model", right_on="Model")
fig, ax = plt.subplots(figsize=(10, 6))
for _, row in merged_hw_metrics.iterrows():
    ax.scatter(row["Latency_ms"], row["Masked_SSIM_mean"], s=250, label=row["model"], edgecolors="black")
    ax.annotate(row["model"], (row["Latency_ms"] + 1.5, row["Masked_SSIM_mean"]), fontsize=10, fontweight="bold")

ax.set_xlabel("Inference Latency per Slice (ms) [Log Scale]", fontsize=12)
ax.set_xscale("log")
ax.set_ylabel("Masked SSIM (Higher is Better)", fontsize=12)
ax.set_title("Clinical Quality vs. Edge Inference Speedup Trade-Off", fontsize=14, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.6)
fig_tradeoff = DIR_FIGURES / "latency_vs_ssim_tradeoff.png"
plt.savefig(fig_tradeoff, dpi=300)
plt.close()
print(f"[FIGURE SAVED] {fig_tradeoff}")

# 3. Find Best Test Slice where Proposed CCR-Diffusion excels
best_slice_name = slice_df[slice_df["model"] == "CCR-Diffusion (Proposed)"].sort_values("Composite_Score").iloc[0]["slice_name"]
best_data = all_qualitative_predictions[best_slice_name]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes[0, 0].imshow(best_data["mri"], cmap="gray")
axes[0, 0].set_title("Input MRI (T1-weighted)", fontweight="bold"); axes[0, 0].axis("off")

axes[0, 1].imshow(best_data["gt"], cmap="gray")
axes[0, 1].set_title("Reference Ground-Truth CT", fontweight="bold"); axes[0, 1].axis("off")

axes[0, 2].imshow(best_data["preds"]["ResUNet Baseline"], cmap="gray")
axes[0, 2].set_title("Model 1: ResUNet Baseline", fontweight="bold"); axes[0, 2].axis("off")

axes[0, 3].imshow(best_data["preds"]["Pix2Pix Generator"], cmap="gray")
axes[0, 3].set_title("Model 2: Pix2Pix Generator", fontweight="bold"); axes[0, 3].axis("off")

axes[1, 0].imshow(best_data["preds"]["cDDPM From Scratch"], cmap="gray")
axes[1, 0].set_title("Model 3: cDDPM From Scratch", fontweight="bold"); axes[1, 0].axis("off")

axes[1, 1].imshow(best_data["preds"]["SOTA-CDDPM (Residual)"], cmap="gray")
axes[1, 1].set_title("Model 4: SOTA-CDDPM (Residual)", fontweight="bold"); axes[1, 1].axis("off")

axes[1, 2].imshow(best_data["preds"]["CCR-Diffusion (Proposed)"], cmap="gray")
axes[1, 2].set_title("Model 5: CCR-Diffusion (Proposed)", fontweight="bold"); axes[1, 2].axis("off")

# Absolute Error Map of CCR-Diffusion
diff_map = np.abs(best_data["preds"]["CCR-Diffusion (Proposed)"] - best_data["gt"])
im_err = axes[1, 3].imshow(diff_map, cmap="inferno")
axes[1, 3].set_title("CCR-Diffusion Absolute Error Map", fontweight="bold"); axes[1, 3].axis("off")
fig.colorbar(im_err, ax=axes[1, 3], fraction=0.046, pad=0.04)

plt.suptitle(f"SynthRAD2023: Qualitative Comparison on Slice {best_slice_name} (Patient {best_data['pid']})", fontsize=16, fontweight="bold")
plt.tight_layout()
fig_qual = DIR_FIGURES / "qualitative_five_model_comparison_deliverable.png"
plt.savefig(fig_qual, dpi=300)
plt.close()
print(f"[FIGURE SAVED] {fig_qual}")

# 4. Zip Deliverables
zip_path = "/kaggle/working/synthrad_five_model_benchmark_deliverables.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, OUTPUT_DIR)
            zipf.write(full_p, arcname=rel_p)

print("\n" + "=" * 80)
print(f"BENCHMARK COMPLETE! Deliverables packaged at:\n{zip_path}")
print("=" * 80)
''')

# Write to file
target_nb_path = "c:/Users/Lohith K/OneDrive/Desktop/training-model/model-resunet/SynthRAD2023_Five_Model_Unified_Benchmark.ipynb"
with open(target_nb_path, "w", encoding="utf-8") as f:
    json.dump({
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }, f, indent=2)

print(f"Successfully generated notebook with {len(cells)} cells at: {target_nb_path}")
