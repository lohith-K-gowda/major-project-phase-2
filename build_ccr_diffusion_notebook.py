# -*- coding: utf-8 -*-
"""
build_ccr_diffusion_notebook.py
Builds the complete, self-contained, publication-ready Jupyter notebook:
SynthRAD_CCR_Diffusion.ipynb
for the SynthRAD2023 Brain MRI-to-CT Synthesis challenge.
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
# CELL 0: MARKDOWN HEADER & SCIENTIFIC RESEARCH ABSTRACT
# ==============================================================================
add_md('''# 🧠 SynthRAD2023: CCR-Diffusion (Coarse-Conditioned Residual Diffusion with Feature Modulation)
## Advanced Multi-Scale FiLM Feature-Modulated MRI-to-Synthetic CT Translation
### Senior Medical AI Research Engineering Deliverable | Kaggle Grandmaster Edition

---

### 🔬 Core Scientific Proposal: "CCR-Diffusion"
Direct end-to-end regression (e.g., standard ResUNet) reliably captures macroscopic brain anatomy and soft-tissue geometry, but suffers from oversmoothing in fine, high-density bony structures. Conversely, standard conditional diffusion models (cDDPMs) that rely on naive channel concatenation force the denoiser to solve both low-frequency anatomical layout and high-frequency structural refinement simultaneously from white noise, leading to slow convergence and hallucination artifacts.

**CCR-Diffusion solves this fundamental dilemma via a principled two-stage hybrid formulation:**
1. **Stage 1 (Anatomical Anchor):** A robust supervised ResUNet maps input MRI to a structural coarse CT estimate ($c = G_\\phi(\\text{MRI})$). Crucially, training is performed under a strict **4-Fold Patient-Level Cross-Fitting** regime, producing strictly **Out-Of-Fold (OOF)** coarse predictions for the 48 training patients with **zero in-sample data leakage**.
2. **Stage 2 (CCR-Diffusion with FiLM Modulation):** Rather than concatenating conditioning images as flat input channels, we embed **Feature-wise Linear Modulation (FiLM)** blocks into the diffusion UNet decoder. Multi-scale feature maps from the Stage-1 ResUNet encoder dynamically scale ($\\gamma$) and shift ($\\beta$) the intermediate feature representations in the diffusion UNet decoder at matching spatial scales ($16\\times 16$, $32\\times 32$, $64\\times 64$, $128\\times 128$).
3. **Residual Formulation:** The diffusion model explicitly denoises the normalized residual $z_0 = (y - c_{\\text{OOF}}) / s$, focusing 100% of its stochastic capacity on fine-grained anatomical details and bone mineralization.
4. **Hybrid Loss Objective:** Diffusion noise MSE across all timesteps ($t \\in [0, 1000]$) combined with a fraction-scaled auxiliary masked SSIM loss computed on estimated clean $x_0$ predictions for low-noise regimes ($t \\le 250$). CT L1 and CT MSE losses are explicitly zeroed out.
5. **Rigorous Validation Standard:** Patient-level validation tracking Masked MAE, Masked SSIM, Masked PSNR, and Composite Score: $\\text{Score} = \\text{mean\\_patient\\_MAE} + 0.25 \\times (1 - \\text{mean\\_patient\\_SSIM})$. The 6 held-out test patients remain completely frozen and isolated.

---

### 📂 Dataset Specification:
- **Cohort:** SynthRAD2023 Task 1 Brain cohort (UMC Utrecht, 60 patients, 3,614 paired 2D slices).
- **Split:** 48 Training Patients (2,879 slices), 6 Validation Patients (372 slices), 6 Test Patients (363 slices).
- **Paths:** `/kaggle/input/datasets/lohithkgowda/synthrad2023-brain-2d-png/synthrad_png/`
- **Working Output:** `/kaggle/working/synthrad_ccr_diffusion/{checkpoints, metrics, figures}`''')

# ==============================================================================
# CELL 1: ENVIRONMENT SETUP, HARDWARE INSPECTION & DETERMINISTIC SEEDING
# ==============================================================================
add_code('''# =====================================================================
# STEP 1: ENVIRONMENT SETUP, HARDWARE INSPECTION & SEED LOCKING
# =====================================================================

import os
import sys
import glob
import math
import time
import json
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

# 1. Hardware Inspection and Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(" SYNTHRAD2023: CCR-DIFFUSION (COARSE-CONDITIONED RESIDUAL DIFFUSION) ")
print("=" * 80)
print(f"PyTorch Version  : {torch.__version__}")
print(f"Compute Device   : {device}")
if torch.cuda.is_available():
    print(f"GPU Model        : {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Total VRAM       : {vram_gb:.2f} GB")
    torch.backends.cudnn.benchmark = True
else:
    print("[WARNING] CUDA not detected. Running on CPU (training will be slow).")

# 2. Strict Deterministic Random Seeding (Seed 42)
SEED = 42
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
print(f"[OK] Deterministic random seed locked to SEED = {SEED}.")
print("=" * 80)''')

# ==============================================================================
# CELL 2: DIRECTORY HIERARCHY, PATH PREFLIGHT & PATIENT SPLIT CHECKS
# ==============================================================================
add_code('''# =====================================================================
# STEP 1 (CONT.): DIRECTORY HIERARCHY & DATASET PREFLIGHT CHECKS
# =====================================================================

# 1. Output Directory Hierarchy under /kaggle/working/synthrad_ccr_diffusion/
BASE_OUT_DIR = "/kaggle/working/synthrad_ccr_diffusion" if os.path.exists("/kaggle") else "./synthrad_ccr_diffusion"
DIR_CHECKPOINTS = os.path.join(BASE_OUT_DIR, "checkpoints")
DIR_METRICS     = os.path.join(BASE_OUT_DIR, "metrics")
DIR_FIGURES     = os.path.join(BASE_OUT_DIR, "figures")
DIR_CACHE       = os.path.join(BASE_OUT_DIR, "cache")

for d in [DIR_CHECKPOINTS, DIR_METRICS, DIR_FIGURES, DIR_CACHE]:
    os.makedirs(d, exist_ok=True)

print(f"[OK] Working output hierarchy created at: {BASE_OUT_DIR}")
print(f"     - Checkpoints : {DIR_CHECKPOINTS}")
print(f"     - Metrics     : {DIR_METRICS}")
print(f"     - Figures     : {DIR_FIGURES}")

# 2. Dynamic Dataset Discovery
CANDIDATE_ROOTS = [
    "/kaggle/input/datasets/lohithkgowda/synthrad2023-brain-2d-png/synthrad_png",
    "/kaggle/input/synthrad2023-brain-2d-png/synthrad_png",
    "./synthrad_png",
    "../synthrad_png"
]

def locate_dataset_root(candidates):
    for path in candidates:
        if os.path.exists(path) and os.path.isdir(os.path.join(path, "ct")) and os.path.isdir(os.path.join(path, "mri")):
            return os.path.abspath(path)
    # Recursive search within /kaggle/input if available
    for match in glob.glob("/kaggle/input/**/synthrad_png", recursive=True):
        if os.path.isdir(os.path.join(match, "ct")) and os.path.isdir(os.path.join(match, "mri")):
            return os.path.abspath(match)
    return None

DATASET_ROOT = locate_dataset_root(CANDIDATE_ROOTS)
if DATASET_ROOT is None:
    raise RuntimeError(
        "CRITICAL ERROR: Could not locate SynthRAD2023 dataset directory!\\n"
        "Please ensure '/kaggle/input/datasets/lohithkgowda/synthrad2023-brain-2d-png/synthrad_png/' "
        "is attached with valid 'ct' and 'mri' subdirectories."
    )

print(f"[OK] SynthRAD2023 Dataset Root: {DATASET_ROOT}")
CT_DIR  = os.path.join(DATASET_ROOT, "ct")
MRI_DIR = os.path.join(DATASET_ROOT, "mri")

all_ct_files  = sorted([f for f in os.listdir(CT_DIR) if f.endswith(".png")])
all_mri_files = sorted([f for f in os.listdir(MRI_DIR) if f.endswith(".png")])
common_files  = sorted(list(set(all_ct_files).intersection(set(all_mri_files))))

if len(common_files) == 0:
    raise RuntimeError("CRITICAL ERROR: No paired CT-MRI files found in dataset directories!")

print(f"[OK] Matched Paired 2D Slices Count: {len(common_files):,}")

# 3. Patient Grouping & Exact Split Preflight Checks
patient_to_slices = {}
for fname in common_files:
    pid = fname.split("_")[0]
    patient_to_slices.setdefault(pid, []).append(fname)

total_patients = len(patient_to_slices)
print(f"[OK] Total unique subjects identified: {total_patients}")

# Exact Fixed Patient Split Lists (48 Train, 6 Val, 6 Test)
TRAIN_PIDS = sorted([
    "1BA001", "1BA005", "1BA012", "1BA014", "1BA022", "1BA032", "1BA040", "1BA058",
    "1BA075", "1BA082", "1BA085", "1BA091", "1BA100", "1BA103", "1BA105", "1BA116",
    "1BA131", "1BA143", "1BA151", "1BA158", "1BA159", "1BA164", "1BA175", "1BA184",
    "1BA185", "1BA189", "1BA201", "1BA206", "1BA220", "1BA222", "1BA227", "1BA239",
    "1BA247", "1BA253", "1BA260", "1BA266", "1BA278", "1BA288", "1BA292", "1BA294",
    "1BA305", "1BA325", "1BA328", "1BA336", "1BA345", "1BA358", "1BA359", "1BA368"
])
VAL_PIDS  = sorted(["1BA076", "1BA111", "1BA125", "1BA141", "1BA300", "1BA379"])
TEST_PIDS = sorted(["1BA054", "1BA097", "1BA172", "1BA234", "1BA256", "1BA307"])

# Assertions on patient sets
discovered_pids = set(patient_to_slices.keys())
missing_train = set(TRAIN_PIDS) - discovered_pids
missing_val   = set(VAL_PIDS) - discovered_pids
missing_test  = set(TEST_PIDS) - discovered_pids

if missing_train or missing_val or missing_test:
    raise RuntimeError(
        f"CRITICAL ERROR: Patient split mismatch!\\n"
        f"Missing Train: {missing_train}\\nMissing Val: {missing_val}\\nMissing Test: {missing_test}"
    )

assert set(TRAIN_PIDS).isdisjoint(set(VAL_PIDS)), "Train and Val patient sets overlap!"
assert set(TRAIN_PIDS).isdisjoint(set(TEST_PIDS)), "Train and Test patient sets overlap!"
assert set(VAL_PIDS).isdisjoint(set(TEST_PIDS)), "Val and Test patient sets overlap!"

def gather_slice_paths(pids):
    mri_paths, ct_paths = [], []
    for pid in pids:
        for sname in sorted(patient_to_slices[pid]):
            mri_paths.append(os.path.join(MRI_DIR, sname))
            ct_paths.append(os.path.join(CT_DIR, sname))
    return mri_paths, ct_paths

train_mri, train_ct = gather_slice_paths(TRAIN_PIDS)
val_mri,   val_ct   = gather_slice_paths(VAL_PIDS)
test_mri,  test_ct  = gather_slice_paths(TEST_PIDS)

print("=" * 80)
print("PATIENT SPLIT INTEGRITY SUMMARY:")
print(f"  - Training Cohort   : {len(TRAIN_PIDS):2d} subjects | {len(train_mri):,} slices (Expected: 2,879)")
print(f"  - Validation Cohort : {len(VAL_PIDS):2d} subjects | {len(val_mri):,} slices (Expected: 372)")
print(f"  - Test Cohort       : {len(TEST_PIDS):2d} subjects | {len(test_mri):,} slices (Expected: 363)")
print("=" * 80)

if len(train_mri) == 0 or len(val_mri) == 0:
    raise RuntimeError("CRITICAL ERROR: Slice loaders or counts are empty!")

print("[PASS] Preflight patient split checks verified with zero patient overlap.")''')

# ==============================================================================
# CELL 3: MASK GENERATION & DATASET PIPELINE
# ==============================================================================
add_code('''# =====================================================================
# STEP 2: PREPROCESSING, BRAIN MASK GENERATION & DATASET CLASSES
# =====================================================================

def extract_brain_body_mask(img_arr, threshold=15.0):
    """
    Derives anatomical brain body mask strictly from MRI to prevent CT leakage.
    - Intensity thresholding at > 15.0 eliminates background scanner noise.
    - Morphological closing with a 7x7 elliptical structuring element seals gaps.
    - Fills the largest external contour to encompass cranial anatomy.
    """
    binary = (img_arr > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img_arr, dtype=np.uint8)
    if contours:
        largest_c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_c], -1, 255, thickness=cv2.FILLED)
    else:
        mask = closed
    return (mask > 0).astype(np.float32)

class SynthRADPairedDataset(Dataset):
    """
    Standard paired dataset loading MRI, CT, and MRI-derived brain mask.
    Normalizes intensities consistently to [-1.0, 1.0].
    Exterior background scanner air is strictly set to -1.0.
    """
    def __init__(self, mri_paths, ct_paths, image_size=128, is_train=False):
        assert len(mri_paths) == len(ct_paths), "Mismatch between MRI and CT paths count"
        self.mri_paths  = mri_paths
        self.ct_paths   = ct_paths
        self.image_size = image_size
        self.is_train   = is_train

    def __len__(self):
        return len(self.mri_paths)

    def _process_slice(self, mri_path, ct_path):
        mri_img = Image.open(mri_path).convert('L')
        ct_img  = Image.open(ct_path).convert('L')
        
        mri_arr = np.array(mri_img, dtype=np.float32)
        ct_arr  = np.array(ct_img,  dtype=np.float32)
        
        # Mask derived strictly from MRI (Zero CT information used)
        mri_mask = extract_brain_body_mask(mri_arr)
        mri_arr  = mri_arr * mri_mask
        
        # Resize: bilinear for images, nearest-neighbor for mask
        mri_res  = cv2.resize(mri_arr,  (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        ct_res   = cv2.resize(ct_arr,   (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(mri_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # Normalization from [0, 255] to [-1.0, 1.0]
        mri_norm = (mri_res / 255.0) * 2.0 - 1.0
        ct_norm  = (ct_res  / 255.0) * 2.0 - 1.0
        
        # Background air clamped strictly to -1.0
        mri_norm[mask_res < 0.5] = -1.0
        ct_norm[mask_res < 0.5]  = -1.0
        
        t_mri  = torch.from_numpy(mri_norm).unsqueeze(0).float()
        t_ct   = torch.from_numpy(ct_norm).unsqueeze(0).float()
        t_mask = torch.from_numpy(mask_res).unsqueeze(0).float()
        
        return t_mri, t_ct, t_mask

    def __getitem__(self, idx):
        mri, ct, mask = self._process_slice(self.mri_paths[idx], self.ct_paths[idx])
        
        if self.is_train:
            # Synchronous horizontal flipping
            if random.random() > 0.5:
                mri  = TF.hflip(mri)
                ct   = TF.hflip(ct)
                mask = TF.hflip(mask)
            # Synchronous mild affine transformations
            angle = random.uniform(-6.0, 6.0)
            tx = int(random.uniform(-3, 3))
            ty = int(random.uniform(-3, 3))
            scale = random.uniform(0.97, 1.03)
            mri  = TF.affine(mri,  angle=angle, translate=[tx, ty], scale=scale, shear=0, fill=-1.0)
            ct   = TF.affine(ct,   angle=angle, translate=[tx, ty], scale=scale, shear=0, fill=-1.0)
            mask = TF.affine(mask, angle=angle, translate=[tx, ty], scale=scale, shear=0, fill=0.0)

        sname = os.path.basename(self.mri_paths[idx])
        pid   = sname.split("_")[0]

        return {
            "mri": mri,
            "ct":  ct,
            "mask": mask,
            "patient_id": pid,
            "slice_name": sname
        }

IMAGE_SIZE = 128
BATCH_SIZE = 8
NUM_WORKERS = 0 if os.name == 'nt' else 2

print(f"[OK] Preprocessing and Paired Dataset pipeline initialized.")''')

# ==============================================================================
# CELL 4: STAGE 1 — RESUNET ARCHITECTURE & SUPERVISED LOSS
# ==============================================================================
add_code('''# =====================================================================
# STEP 3: STAGE-1 RESUNET (ANATOMICAL ANCHOR) ARCHITECTURE
# =====================================================================

class ResBlock(nn.Module):
    """Residual convolutional block with GroupNorm, LeakyReLU, and 1x1 shortcut."""
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

class Stage1ResUNet(nn.Module):
    """
    Stage-1 Anatomical Anchor ResUNet:
    - Supervised regression mapping MRI to structural Coarse CT in [-1, 1].
    - Contains extract_encoder_features() to feed multi-scale conditioning
      feature maps directly into Stage-2 FiLM modulation blocks.
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()
        f = base_filters
        self.enc1  = ResBlock(in_channels, f)        # 128x128, 32
        self.pool1 = nn.MaxPool2d(2)                # 64x64
        self.enc2  = ResBlock(f, f * 2)              # 64x64, 64
        self.pool2 = nn.MaxPool2d(2)                # 32x32
        self.enc3  = ResBlock(f * 2, f * 4)          # 32x32, 128
        self.pool3 = nn.MaxPool2d(2)                # 16x16
        self.enc4  = ResBlock(f * 4, f * 8)          # 16x16, 256
        self.pool4 = nn.MaxPool2d(2)                # 8x8

        self.bottleneck = nn.Sequential(
            ResBlock(f * 8, f * 8),
            ResBlock(f * 8, f * 8)
        )

        self.up4  = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec4 = ResBlock(f * 8 + f * 4, f * 4)
        self.up3  = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec3 = ResBlock(f * 4 + f * 2, f * 2)
        self.up2  = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec2 = ResBlock(f * 2 + f, f)
        self.up1  = nn.ConvTranspose2d(f, f, kernel_size=2, stride=2)
        self.dec1 = ResBlock(f + f, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)
        self.final_act  = nn.Tanh()

    def extract_encoder_features(self, x):
        """Extracts multi-scale encoder feature representations for FiLM."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        return {
            "enc1": e1,  # [B, 32, 128, 128]
            "enc2": e2,  # [B, 64, 64, 64]
            "enc3": e3,  # [B, 128, 32, 32]
            "enc4": e4,  # [B, 256, 16, 16]
            "bot":  b    # [B, 256, 8, 8]
        }

    def forward(self, x):
        feats = self.extract_encoder_features(x)
        e1, e2, e3, e4, b = feats["enc1"], feats["enc2"], feats["enc3"], feats["enc4"], feats["bot"]
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final_act(self.final_conv(d1))

class DifferentiableSSIM(nn.Module):
    """Differentiable masked SSIM loss for normalized [-1.0, 1.0] domain."""
    def __init__(self, window_size=11, sigma=1.5, data_range=2.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        _2d = torch.outer(g, g).unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", _2d)

    def forward(self, img1, img2, mask=None):
        w = self.window.to(dtype=img1.dtype, device=img1.device)
        pad = self.window_size // 2
        mu1 = F.conv2d(img1, w, padding=pad)
        mu2 = F.conv2d(img2, w, padding=pad)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, w, padding=pad) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, w, padding=pad) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, w, padding=pad) - mu1_mu2
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
        )
        if mask is not None:
            m = mask.to(dtype=img1.dtype, device=img1.device)
            return ((1.0 - ssim_map) * m).sum() / (m.sum() + 1e-8)
        return 1.0 - ssim_map.mean()

stage1_demo = Stage1ResUNet().to(device)
s1_params = sum(p.numel() for p in stage1_demo.parameters() if p.requires_grad)
print(f"[OK] Stage-1 ResUNet initialized: {s1_params:,} parameters (~{s1_params/1e6:.2f}M).")''')

# ==============================================================================
# CELL 5: 4-FOLD CROSS-FITTING & OOF COARSE CACHING
# ==============================================================================
add_code('''# =====================================================================
# STEP 3 (CONT.): 4-FOLD CROSS-FITTING & LEAKAGE-FREE OOF CACHING
# =====================================================================

N_FOLDS = 4
PATIENTS_PER_FOLD = 12

FOLDS = {}
for k in range(N_FOLDS):
    fold_pids = TRAIN_PIDS[k * PATIENTS_PER_FOLD : (k + 1) * PATIENTS_PER_FOLD]
    FOLDS[k + 1] = fold_pids

print("=" * 80)
print("PATIENT-LEVEL 4-FOLD CROSS-FITTING PARTITIONS:")
print("=" * 80)
for k, pids in FOLDS.items():
    m_paths, _ = gather_slice_paths(pids)
    print(f"  Fold {k}: {len(pids)} patients ({pids[0]} ... {pids[-1]}), {len(m_paths):,} slices")
print("=" * 80)

DIR_CKPT_COARSE = os.path.join(DIR_CHECKPOINTS, "coarse_resunet")
os.makedirs(DIR_CKPT_COARSE, exist_ok=True)

COARSE_EPOCHS = 35
COARSE_LR     = 1e-4

def train_coarse_fold(fold_idx, train_pids_fold, val_pids_fold):
    """Trains a ResUNet on 3 folds and saves the best model evaluated on the held-out fold."""
    ckpt_path = os.path.join(DIR_CKPT_COARSE, f"coarse_resunet_fold_{fold_idx}.pth")
    if os.path.exists(ckpt_path):
        print(f"[FOUND] Existing checkpoint for Fold {fold_idx}: {ckpt_path}. Skipping retraining.")
        model = Stage1ResUNet().to(device)
        data = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(data["model_state_dict"])
        return model

    print(f"\\n--> Training Coarse ResUNet Fold {fold_idx}/{N_FOLDS} ({len(train_pids_fold)} Train PIDs, {len(val_pids_fold)} Val PIDs)...")
    tr_m, tr_c = gather_slice_paths(train_pids_fold)
    va_m, va_c = gather_slice_paths(val_pids_fold)
    
    ds_tr = SynthRADPairedDataset(tr_m, tr_c, image_size=IMAGE_SIZE, is_train=True)
    ds_va = SynthRADPairedDataset(va_m, va_c, image_size=IMAGE_SIZE, is_train=False)
    
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = Stage1ResUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=COARSE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=COARSE_EPOCHS, eta_min=1e-6)
    ssim_crit = DifferentiableSSIM(window_size=11, data_range=2.0)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_val_mae = float("inf")
    best_epoch = 0

    for epoch in range(1, COARSE_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch in dl_tr:
            mri  = batch["mri"].to(device)
            ct   = batch["ct"].to(device)
            mask = batch["mask"].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                pred = model(mri)
                diff = torch.abs(pred - ct)
                loss_l1 = (diff * mask).sum() / (mask.sum() + 1e-8)
                loss_ssim = ssim_crit(pred, ct, mask=mask)
                loss = loss_l1 + 0.10 * loss_ssim
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            
        scheduler.step()
        ep_tr_loss = running_loss / len(dl_tr)

        # Validation on held-out fold
        model.eval()
        running_val_l1 = 0.0
        with torch.no_grad():
            for batch in dl_va:
                mri  = batch["mri"].to(device)
                ct   = batch["ct"].to(device)
                mask = batch["mask"].to(device)
                pred = model(mri)
                diff = torch.abs(pred - ct)
                running_val_l1 += ((diff * mask).sum() / (mask.sum() + 1e-8)).item()
                
        ep_val_l1 = running_val_l1 / len(dl_va)
        if ep_val_l1 < best_val_mae:
            best_val_mae = ep_val_l1
            best_epoch = epoch
            torch.save({
                "fold": fold_idx,
                "epoch": best_epoch,
                "best_val_mae": best_val_mae,
                "model_state_dict": model.state_dict()
            }, ckpt_path)

        if epoch % 10 == 0 or epoch == COARSE_EPOCHS:
            print(f"  Fold {fold_idx} | Ep [{epoch:02d}/{COARSE_EPOCHS}] | Train Loss: {ep_tr_loss:.4f} | Val L1: {ep_val_l1:.4f} (Best: {best_val_mae:.4f} @ Ep {best_epoch})")

    best_data = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_data["model_state_dict"])
    return model

# Train or load the 4 fold models
COARSE_MODELS = {}
for k in range(1, N_FOLDS + 1):
    val_pids_k = FOLDS[k]
    train_pids_k = [pid for j, pids in FOLDS.items() if j != k for pid in pids]
    COARSE_MODELS[k] = train_coarse_fold(k, train_pids_k, val_pids_k)

print("\\n[OK] All 4 Stage-1 ResUNet fold models ready.")

# Generate and cache OOF predictions for training set, and ensemble predictions for validation set
OOF_CACHE_FILE = os.path.join(DIR_CACHE, "oof_coarse_predictions.pt")
VAL_CACHE_FILE = os.path.join(DIR_CACHE, "val_coarse_predictions.pt")

if os.path.exists(OOF_CACHE_FILE) and os.path.exists(VAL_CACHE_FILE):
    print(f"[FOUND] Loading cached coarse predictions from disk...")
    OOF_COARSE_DICT = torch.load(OOF_CACHE_FILE, map_location="cpu")
    VAL_COARSE_DICT = torch.load(VAL_CACHE_FILE, map_location="cpu")
else:
    print("\\n--> Generating zero-leakage Out-Of-Fold (OOF) training coarse predictions...")
    OOF_COARSE_DICT = {}
    for k in range(1, N_FOLDS + 1):
        held_out_pids = FOLDS[k]
        m_paths, c_paths = gather_slice_paths(held_out_pids)
        ds_k = SynthRADPairedDataset(m_paths, c_paths, image_size=IMAGE_SIZE, is_train=False)
        dl_k = DataLoader(ds_k, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        model_k = COARSE_MODELS[k].to(device)
        model_k.eval()
        with torch.no_grad():
            for batch in tqdm(dl_k, desc=f"Predicting OOF Fold {k}/{N_FOLDS}"):
                mri    = batch["mri"].to(device)
                mask   = batch["mask"].to(device)
                snames = batch["slice_name"]
                pred_c = model_k(mri)
                pred_c = torch.where(mask > 0.5, pred_c, torch.full_like(pred_c, -1.0))
                for i in range(mri.size(0)):
                    OOF_COARSE_DICT[snames[i]] = pred_c[i].cpu()

    torch.save(OOF_COARSE_DICT, OOF_CACHE_FILE)
    print(f"[OK] Cached {len(OOF_COARSE_DICT):,} OOF coarse predictions to {OOF_CACHE_FILE}")

    print("\\n--> Generating deterministic ensemble coarse predictions for validation set...")
    VAL_COARSE_DICT = {}
    ds_val = SynthRADPairedDataset(val_mri, val_ct, image_size=IMAGE_SIZE, is_train=False)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    with torch.no_grad():
        for batch in tqdm(dl_val, desc="Predicting Ensemble Val Coarse"):
            mri    = batch["mri"].to(device)
            mask   = batch["mask"].to(device)
            snames = batch["slice_name"]
            
            # 4-model ensemble mean
            ens_preds = []
            for k in range(1, N_FOLDS + 1):
                c_k = COARSE_MODELS[k](mri)
                ens_preds.append(c_k)
            c_ens = torch.stack(ens_preds, dim=0).mean(dim=0)
            c_ens = torch.where(mask > 0.5, c_ens, torch.full_like(c_ens, -1.0))
            for i in range(mri.size(0)):
                VAL_COARSE_DICT[snames[i]] = c_ens[i].cpu()

    torch.save(VAL_COARSE_DICT, VAL_CACHE_FILE)
    print(f"[OK] Cached {len(VAL_COARSE_DICT):,} Validation coarse predictions to {VAL_CACHE_FILE}")

# Calculate fixed residual standard deviation scale s
train_ds_eval = SynthRADPairedDataset(train_mri, train_ct, image_size=IMAGE_SIZE, is_train=False)
train_eval_loader = DataLoader(train_ds_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

residual_masked_values = []
for batch in tqdm(train_eval_loader, desc="Computing OOF Residual Scale s"):
    ct     = batch["ct"]
    mask   = batch["mask"]
    snames = batch["slice_name"]
    for i in range(ct.size(0)):
        sname = snames[i]
        c_pred = OOF_COARSE_DICT[sname]
        gt_ct  = ct[i]
        m_arr  = mask[i]
        r = gt_ct - c_pred
        body = m_arr > 0.5
        if body.sum() > 0:
            residual_masked_values.extend(r[body].numpy().tolist())

res_np = np.array(residual_masked_values, dtype=np.float32)
RESIDUAL_SCALE = float(np.std(res_np))
print("=" * 80)
print(f"OUT-OF-FOLD RESIDUAL STATISTICS:")
print(f"  - Mean Residual           : {np.mean(res_np):.5f}")
print(f"  - Residual Std Dev (Scale): {RESIDUAL_SCALE:.5f}  <-- Fixed Residual Scale s")
print("=" * 80)''')

# ==============================================================================
# CELL 6: STAGE 2 — CCR-DIFFUSION WITH FILM MODULATION ARCHITECTURE
# ==============================================================================
add_code('''# =====================================================================
# STEP 4: STAGE-2 CCR-DIFFUSION WITH NOVEL FILM MODULATION
# =====================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Standard sinusoidal timestep embeddings for diffusion denoiser."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000.0) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer:
    Modulates intermediate decoder features F via learned scale gamma and shift beta:
    FiLM(F; gamma, beta) = (1.0 + gamma) * F + beta
    Modulation parameters are derived dynamically from conditioning feature maps.
    Weights are initialized to zero so the layer acts as an identity operator at initialization.
    """
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
    """
    Diffusion Residual Block integrating:
    1. Timestep embedding injection
    2. GroupNorm & SiLU activations
    3. Multi-scale FiLM modulation from Stage-1 ResUNet encoder features
    """
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

        if cond_channels is not None:
            self.film = FiLMBlock(out_ch, cond_channels)
        else:
            self.film = None

    def forward(self, x, t_emb, cond_feat=None):
        h = self.act1(self.gn1(self.conv1(x)))
        h = h + self.act1(self.time_mlp(t_emb))[:, :, None, None]
        h = self.act2(self.gn2(self.conv2(h)))
        out = h + self.shortcut(x)
        if self.film is not None and cond_feat is not None:
            out = self.film(out, cond_feat)
        return out

class ConditionalDiffusionUNet(nn.Module):
    """
    Stage-2 CCR-Diffusion UNet with Multi-Scale FiLM Feature Modulation:
    - Input: Noisy residual z_t + Brain mask [B, 2, 128, 128]
    - Output: Predicted Gaussian noise epsilon [B, 1, 128, 128]
    - FiLM Decoder Modulation: Decoder stages are dynamically modulated by Stage-1
      multi-scale encoder features at corresponding resolutions:
      * enc4 (16x16, 256 ch) modulates dec4
      * enc3 (32x32, 128 ch) modulates dec3
      * enc2 (64x64, 64 ch) modulates dec2
      * enc1 (128x128, 32 ch) modulates dec1
    """
    def __init__(self, in_channels=2, out_channels=1, base_channels=32, time_emb_dim=64,
                 cond_channels=(32, 64, 128, 256)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        c1, c2, c3, c4 = cond_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        c = base_channels

        # Encoder stages (Processing noisy residual z_t and mask)
        self.enc1  = DiffusionResBlockFiLM(in_channels, c, time_emb_dim)            # 128x128
        self.down1 = nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1)            # 64x64
        self.enc2  = DiffusionResBlockFiLM(c, c * 2, time_emb_dim)                  # 64x64
        self.down2 = nn.Conv2d(c * 2, c * 2, kernel_size=3, stride=2, padding=1)    # 32x32
        self.enc3  = DiffusionResBlockFiLM(c * 2, c * 4, time_emb_dim)              # 32x32
        self.down3 = nn.Conv2d(c * 4, c * 4, kernel_size=3, stride=2, padding=1)    # 16x16
        self.enc4  = DiffusionResBlockFiLM(c * 4, c * 8, time_emb_dim)              # 16x16
        self.down4 = nn.Conv2d(c * 8, c * 8, kernel_size=3, stride=2, padding=1)    # 8x8

        # Bottleneck
        self.bot1  = DiffusionResBlockFiLM(c * 8, c * 8, time_emb_dim)              # 8x8
        self.bot2  = DiffusionResBlockFiLM(c * 8, c * 8, time_emb_dim)              # 8x8

        # Decoder stages with skip concatenation and FiLM modulation
        self.up4   = nn.ConvTranspose2d(c * 8, c * 8, kernel_size=2, stride=2)      # 16x16
        self.dec4  = DiffusionResBlockFiLM(c * 8 + c * 8, c * 4, time_emb_dim, cond_channels=c4)

        self.up3   = nn.ConvTranspose2d(c * 4, c * 4, kernel_size=2, stride=2)      # 32x32
        self.dec3  = DiffusionResBlockFiLM(c * 4 + c * 4, c * 2, time_emb_dim, cond_channels=c3)

        self.up2   = nn.ConvTranspose2d(c * 2, c * 2, kernel_size=2, stride=2)      # 64x64
        self.dec2  = DiffusionResBlockFiLM(c * 2 + c * 2, c, time_emb_dim, cond_channels=c2)

        self.up1   = nn.ConvTranspose2d(c, c, kernel_size=2, stride=2)              # 128x128
        self.dec1  = DiffusionResBlockFiLM(c + c, c, time_emb_dim, cond_channels=c1)

        self.final_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, z_t, t, cond_features):
        t_emb = self.time_mlp(t)

        # Forward encoder
        e1 = self.enc1(z_t, t_emb)
        d1 = self.down1(e1)
        e2 = self.enc2(d1, t_emb)
        d2 = self.down2(e2)
        e3 = self.enc3(d2, t_emb)
        d3 = self.down3(e3)
        e4 = self.enc4(d3, t_emb)
        d4 = self.down4(e4)

        # Bottleneck
        b = self.bot2(self.bot1(d4, t_emb), t_emb)

        # Forward decoder with FiLM modulation
        u4 = self.up4(b)
        d4_out = self.dec4(torch.cat([u4, e4], dim=1), t_emb, cond_feat=cond_features["enc4"])

        u3 = self.up3(d4_out)
        d3_out = self.dec3(torch.cat([u3, e3], dim=1), t_emb, cond_feat=cond_features["enc3"])

        u2 = self.up2(d3_out)
        d2_out = self.dec2(torch.cat([u2, e2], dim=1), t_emb, cond_feat=cond_features["enc2"])

        u1 = self.up1(d2_out)
        d1_out = self.dec1(torch.cat([u1, e1], dim=1), t_emb, cond_feat=cond_features["enc1"])

        return self.final_conv(d1_out)

ccr_diffusion_model = ConditionalDiffusionUNet().to(device)
diff_params = sum(p.numel() for p in ccr_diffusion_model.parameters() if p.requires_grad)
print("=" * 80)
print(f"STAGE-2 CCR-DIFFUSION WITH FILM MODULATION INITIALIZED")
print(f"  Trainable Parameters : {diff_params:,} (~{diff_params/1e6:.2f}M)")
print(f"  Inputs               : [z_t, brain_mask] (2 channels)")
print(f"  Conditioning Hook    : Multi-scale Stage-1 ResUNet encoder feature maps")
print(f"  Outputs              : Predicted Gaussian noise epsilon (1 channel)")
print("=" * 80)''')

# ==============================================================================
# CELL 7: DIFFUSION SCHEDULER & DETERMINISTIC DDIM REVERSE SAMPLER
# ==============================================================================
add_code('''# =====================================================================
# STEP 4 (CONT.): DDPM NOISE SCHEDULE & DETERMINISTIC DDIM SAMPLER
# =====================================================================

class ResidualDDPMScheduler:
    """
    Standard DDPM noise scheduler for normalized residuals z_0 = (y - c) / s.
    Forward process: q_sample(z_0, t, noise) = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * noise
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.timesteps = timesteps
        self.device = torch.device(device) if isinstance(device, str) else device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

    def q_sample(self, z_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z_0)
        alphas_cp = self.alphas_cumprod.to(z_0.device)
        sqrt_alpha = alphas_cp[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1.0 - alphas_cp[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha * z_0 + sqrt_one_minus * noise, noise

    def reconstruct_z0(self, z_t, noise, t):
        """Algebraic inversion to estimate clean residual z_0 from noisy z_t and predicted noise."""
        alphas_cp = self.alphas_cumprod.to(z_t.device)
        sqrt_alpha = alphas_cp[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1.0 - alphas_cp[t]).sqrt().view(-1, 1, 1, 1)
        return (z_t - sqrt_one_minus * noise) / (sqrt_alpha + 1e-8)

sched = ResidualDDPMScheduler(timesteps=1000, beta_start=1e-4, beta_end=0.02, device=device)

# Mathematical Preflight Verification
dummy_z0 = torch.randn(2, 1, 128, 128, device=device)
dummy_t  = torch.tensor([50, 500], device=device).long()
z_t, true_noise = sched.q_sample(dummy_z0, dummy_t)
z0_rec = sched.reconstruct_z0(z_t, true_noise, dummy_t)
inversion_err = (z0_rec - dummy_z0).abs().max().item()
assert inversion_err < 1e-4, f"Algebraic inversion error too large: {inversion_err}"
print(f"[PASS] DDPM forward schedule and algebraic inversion verified (error < 1e-4).")

@torch.no_grad()
def sample_ccr_diffusion_ddim(diff_model, stage1_model, mri, coarse_ct, mask,
                              residual_scale=RESIDUAL_SCALE, ddim_steps=50, initial_noise=None):
    """
    Deterministic DDIM reverse sampling (eta = 0.0) for CCR-Diffusion:
    1. Extracts multi-scale conditioning features from Stage-1 ResUNet.
    2. Runs 50 deterministic DDIM reverse steps.
    3. Reconstructs synthetic CT: sCT = clip(coarse_ct + s * z_0, -1.0, 1.0).
    4. Enforces background scanner air outside brain mask strictly to -1.0.
    """
    diff_model.eval()
    stage1_model.eval()
    dev = mri.device
    b   = mri.shape[0]

    # Extract Stage-1 encoder features for FiLM modulation
    cond_features = stage1_model.extract_encoder_features(mri)

    # Initialize latent noise z_T
    if initial_noise is not None:
        z_t = initial_noise.to(dev).clone()
    else:
        z_t = torch.randn((b, 1, IMAGE_SIZE, IMAGE_SIZE), device=dev)

    timesteps = sorted(torch.linspace(0, sched.timesteps - 1, ddim_steps).long().tolist(), reverse=True)
    alphas_cp = sched.alphas_cumprod.to(dev)

    for i, t_val in enumerate(timesteps):
        t = torch.full((b,), t_val, device=dev, dtype=torch.long)
        
        # Denoiser input: [z_t, mask] (2 channels)
        model_in = torch.cat([z_t, mask], dim=1)
        pred_noise = diff_model(model_in, t, cond_features)

        alpha_cur  = alphas_cp[t_val]
        alpha_prev = alphas_cp[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=dev)

        # Estimated clean residual z_0 (residuals are unconstrained, bounded only against numerical drift)
        pred_z0 = (z_t - (1.0 - alpha_cur).sqrt() * pred_noise) / (alpha_cur.sqrt() + 1e-8)
        pred_z0 = torch.clamp(pred_z0, -6.0, 6.0)

        # DDIM reverse step (eta = 0.0)
        pred_noise_adj = (z_t - alpha_cur.sqrt() * pred_z0) / ((1.0 - alpha_cur).sqrt() + 1e-8)
        dir_zt = (1.0 - alpha_prev).clamp(min=0.0).sqrt() * pred_noise_adj
        z_t = alpha_prev.sqrt() * pred_z0 + dir_zt

    # Final CT reconstruction: sCT = clip(Coarse + s * z_0, -1.0, 1.0)
    pred_residual = z_t * residual_scale
    sct = torch.clamp(coarse_ct + pred_residual, -1.0, 1.0)
    
    # Air exterior restored strictly to -1.0
    sct = torch.where(mask > 0.5, sct, torch.full_like(sct, -1.0))

    return sct, pred_residual

print("[OK] Deterministic DDIM reverse sampler initialized.")''')

# ==============================================================================
# CELL 8: STAGE 2 DATASET & DATA LOADERS
# ==============================================================================
add_code('''# =====================================================================
# STEP 2 (CONT.): STAGE-2 DATASET & ROBUST DATA LOADERS
# =====================================================================

class SynthRADStage2CCRDataset(Dataset):
    """
    Dataset for Stage-2 CCR-Diffusion training:
    Loads MRI, CT, Mask and matches with pre-cached Out-Of-Fold Coarse CT predictions.
    Target clean residual: z_0 = (CT - Coarse_OOF) / s.
    """
    def __init__(self, mri_paths, ct_paths, coarse_cache, residual_scale, image_size=128, is_train=True):
        self.mri_paths    = mri_paths
        self.ct_paths     = ct_paths
        self.coarse_cache = coarse_cache
        self.scale        = residual_scale
        self.image_size   = image_size
        self.is_train     = is_train

    def __len__(self):
        return len(self.mri_paths)

    def __getitem__(self, idx):
        mri_path = self.mri_paths[idx]
        ct_path  = self.ct_paths[idx]
        sname    = os.path.basename(mri_path)

        mri_img = Image.open(mri_path).convert('L')
        ct_img  = Image.open(ct_path).convert('L')
        
        mri_arr = np.array(mri_img, dtype=np.float32)
        ct_arr  = np.array(ct_img,  dtype=np.float32)
        
        mri_mask = extract_brain_body_mask(mri_arr)
        mri_arr  = mri_arr * mri_mask
        
        mri_res  = cv2.resize(mri_arr,  (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        ct_res   = cv2.resize(ct_arr,   (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(mri_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        mri_norm = (mri_res / 255.0) * 2.0 - 1.0
        ct_norm  = (ct_res  / 255.0) * 2.0 - 1.0
        
        mri_norm[mask_res < 0.5] = -1.0
        ct_norm[mask_res < 0.5]  = -1.0
        
        t_mri  = torch.from_numpy(mri_norm).unsqueeze(0).float()
        t_ct   = torch.from_numpy(ct_norm).unsqueeze(0).float()
        t_mask = torch.from_numpy(mask_res).unsqueeze(0).float()
        
        # Pre-cached coarse prediction
        t_coarse = self.coarse_cache[sname] # [1, H, W]

        # Target normalized residual: z_0 = (CT - Coarse) / s
        r = t_ct - t_coarse
        z_0 = r / self.scale

        return {
            "mri": t_mri,
            "ct": t_ct,
            "coarse": t_coarse,
            "mask": t_mask,
            "z_0": z_0,
            "slice_name": sname,
            "patient_id": sname.split("_")[0]
        }

# Instantiate Stage-2 Data Loaders
train_dataset_stage2 = SynthRADStage2CCRDataset(
    train_mri, train_ct, OOF_COARSE_DICT, RESIDUAL_SCALE, image_size=IMAGE_SIZE, is_train=True
)
train_loader_stage2 = DataLoader(
    train_dataset_stage2, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
)

val_metric_dataset = SynthRADStage2CCRDataset(
    val_mri, val_ct, VAL_COARSE_DICT, RESIDUAL_SCALE, image_size=IMAGE_SIZE, is_train=False
)
val_metric_loader = DataLoader(
    val_metric_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)

val_visualization_loader = DataLoader(
    val_metric_dataset, batch_size=4, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)

print("=" * 80)
print(f"STAGE-2 CCR-DIFFUSION DATA LOADERS INITIALIZED:")
print(f"  - train_loader_stage2      : {len(train_loader_stage2)} batches ({len(train_dataset_stage2)} slices)")
print(f"  - val_metric_loader        : {len(val_metric_loader)} batches ({len(val_metric_dataset)} slices)")
print(f"  - val_visualization_loader : {len(val_visualization_loader)} batches ({len(val_metric_dataset)} slices)")
print("=" * 80)''')

# ==============================================================================
# CELL 9: HYBRID LOSS OBJECTIVES & VALIDATION METRIC SUITE
# ==============================================================================
add_code('''# =====================================================================
# STEP 5 & 6: HYBRID LOSS OBJECTIVES & PATIENT-LEVEL VALIDATION
# =====================================================================

class CCRHybridLoss(nn.Module):
    """
    Hybrid Training Loss Function for CCR-Diffusion:
    1. Masked Noise-Prediction MSE across all timesteps (t in [0, 1000]).
    2. Fraction-scaled auxiliary Masked SSIM loss calculated on estimated clean x_0
       for lower-noise timesteps (t <= 250).
    3. Explicitly zeroed out CT L1 and CT MSE losses (rely strictly on noise MSE + auxiliary SSIM).
    """
    def __init__(self, lambda_aux_ssim=0.05, ssim_cutoff_t=250):
        super().__init__()
        self.lambda_aux_ssim = lambda_aux_ssim
        self.ssim_cutoff_t = ssim_cutoff_t
        self.ssim_criterion = DifferentiableSSIM(window_size=11, data_range=2.0)
        
        # Explicit zeroing of CT L1 and CT MSE losses as required
        self.lambda_ct_l1  = 0.0
        self.lambda_ct_mse = 0.0

    def forward(self, pred_noise, true_noise, mask, t, z_t, coarse, target_ct, residual_scale, scheduler):
        # 1. Masked Noise MSE Loss across all timesteps
        diff_sq = (pred_noise - true_noise) ** 2
        m_sum = mask.sum() + 1e-8
        loss_noise_mse = (diff_sq * mask).sum() / m_sum

        # 2. Auxiliary Masked SSIM Loss on estimated clean x_0 for low-noise timesteps
        low_noise_mask = (t <= self.ssim_cutoff_t)
        if low_noise_mask.sum() > 0:
            sub_zt     = z_t[low_noise_mask]
            sub_noise  = pred_noise[low_noise_mask]
            sub_t      = t[low_noise_mask]
            sub_coarse = coarse[low_noise_mask]
            sub_ct     = target_ct[low_noise_mask]
            sub_mask   = mask[low_noise_mask]

            # Reconstruct estimated clean residual z_0 and clean CT x_0
            pred_z0 = scheduler.reconstruct_z0(sub_zt, sub_noise, sub_t)
            pred_ct_x0 = torch.clamp(sub_coarse + pred_z0 * residual_scale, -1.0, 1.0)

            loss_aux_ssim = self.ssim_criterion(pred_ct_x0, sub_ct, mask=sub_mask)
        else:
            loss_aux_ssim = torch.tensor(0.0, device=pred_noise.device)

        total_loss = loss_noise_mse + self.lambda_aux_ssim * loss_aux_ssim
        return total_loss, loss_noise_mse, loss_aux_ssim

ccr_loss_engine = CCRHybridLoss(lambda_aux_ssim=0.05, ssim_cutoff_t=250)

# Patient-Level Validation Metric Suite
def evaluate_slice_metrics(pred_norm, gt_norm, mask_arr):
    pred_c = np.clip(pred_norm, -1.0, 1.0)
    gt_c   = np.clip(gt_norm,   -1.0, 1.0)
    pred_u = (pred_c + 1.0) / 2.0
    gt_u   = (gt_c   + 1.0) / 2.0

    body = mask_arr > 0.5
    if body.sum() > 0:
        masked_mae = float(np.mean(np.abs(pred_u[body] - gt_u[body])))
        masked_mse = float(np.mean((pred_u[body] - gt_u[body]) ** 2))
        masked_psnr = float(10.0 * np.log10(1.0 / (masked_mse + 1e-8)))
        _, ssim_map = compute_ssim(gt_u, pred_u, data_range=1.0, full=True)
        masked_ssim = float(np.mean(ssim_map[body]))
    else:
        masked_mae = float(np.mean(np.abs(pred_u - gt_u)))
        masked_mse = float(np.mean((pred_u - gt_u) ** 2))
        masked_psnr = float(10.0 * np.log10(1.0 / (masked_mse + 1e-8)))
        masked_ssim = float(compute_ssim(gt_u, pred_u, data_range=1.0))

    return {
        "Masked_MAE": masked_mae,
        "Masked_SSIM": masked_ssim,
        "Masked_PSNR": masked_psnr
    }

# Fixed repeatable validation noise pool
gen_val = torch.Generator().manual_seed(SEED)
FIXED_VAL_NOISE_LIST = [
    torch.randn((1, 1, IMAGE_SIZE, IMAGE_SIZE), generator=gen_val) for _ in range(len(val_metric_dataset))
]

def evaluate_patient_level_validation(diff_model, stage1_model, ddim_steps=50):
    """
    Computes patient-level metrics across the frozen validation cohort:
    - Patient Mean Masked MAE
    - Patient Mean Masked SSIM
    - Patient Mean Masked PSNR
    - Composite Score: mean_patient_MAE + 0.25 * (1 - mean_patient_SSIM)
    """
    diff_model.eval()
    stage1_model.eval()

    patient_records = {}
    idx_counter = 0

    with torch.no_grad():
        for batch in val_metric_loader:
            mri    = batch["mri"].to(device)
            ct     = batch["ct"].to(device)
            coarse = batch["coarse"].to(device)
            mask   = batch["mask"].to(device)
            pids   = batch["patient_id"]
            b_sz   = mri.size(0)

            b_noise = torch.cat([FIXED_VAL_NOISE_LIST[idx_counter + i] for i in range(b_sz)], dim=0).to(device)
            idx_counter += b_sz

            sct, _ = sample_ccr_diffusion_ddim(
                diff_model, stage1_model, mri, coarse, mask,
                residual_scale=RESIDUAL_SCALE, ddim_steps=ddim_steps, initial_noise=b_noise
            )

            sct_np  = sct.squeeze(1).cpu().numpy()
            ct_np   = ct.squeeze(1).cpu().numpy()
            mask_np = mask.squeeze(1).cpu().numpy()

            for i in range(b_sz):
                res = evaluate_slice_metrics(sct_np[i], ct_np[i], mask_np[i])
                patient_records.setdefault(pids[i], []).append(res)

    patient_summary = []
    for pid, records in patient_records.items():
        pdf = pd.DataFrame(records)
        patient_summary.append({
            "patient_id": pid,
            "patient_mae": float(pdf["Masked_MAE"].mean()),
            "patient_ssim": float(pdf["Masked_SSIM"].mean()),
            "patient_psnr": float(pdf["Masked_PSNR"].mean())
        })

    df_p = pd.DataFrame(patient_summary)
    mean_patient_mae  = float(df_p["patient_mae"].mean())
    mean_patient_ssim = float(df_p["patient_ssim"].mean())
    mean_patient_psnr = float(df_p["patient_psnr"].mean())
    
    # Exact composite score formula required
    composite_score = mean_patient_mae + 0.25 * (1.0 - mean_patient_ssim)

    return composite_score, mean_patient_mae, mean_patient_ssim, mean_patient_psnr, df_p

print("[OK] Hybrid Loss and Patient-Level Validation Evaluation Suite ready.")''')

# ==============================================================================
# CELL 10: AUTOMATED VISUALIZATION GRID FUNCTION
# ==============================================================================
add_code('''# =====================================================================
# STEP 6 (CONT.): AUTOMATED GRID VISUALIZATION BLOCK (EVERY 10 EPOCHS)
# =====================================================================

def render_visualization_grid(diff_model, stage1_model, epoch, save_dir=DIR_FIGURES):
    """
    Automated grid visualization displaying:
    [Input MRI | Ground-Truth CT | Stage-1 Coarse CT | Stage-2 CCR-Diffusion CT | Masked Absolute Error]
    """
    diff_model.eval()
    stage1_model.eval()

    sample_batch = next(iter(val_visualization_loader))
    mri_b    = sample_batch["mri"][:2].to(device)
    ct_b     = sample_batch["ct"][:2].to(device)
    coarse_b = sample_batch["coarse"][:2].to(device)
    mask_b   = sample_batch["mask"][:2].to(device)
    pids_b   = sample_batch["patient_id"][:2]

    with torch.no_grad():
        sct_b, _ = sample_ccr_diffusion_ddim(
            diff_model, stage1_model, mri_b, coarse_b, mask_b,
            residual_scale=RESIDUAL_SCALE, ddim_steps=50
        )

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), dpi=120)
    for row in range(2):
        m_img = mri_b[row, 0].cpu().numpy()
        g_img = ct_b[row, 0].cpu().numpy()
        c_img = coarse_b[row, 0].cpu().numpy()
        s_img = sct_b[row, 0].cpu().numpy()
        m_msk = mask_b[row, 0].cpu().numpy()
        
        # Masked absolute error
        diff = np.abs(s_img - g_img) * (m_msk > 0.5)

        axes[row, 0].imshow(m_img, cmap="gray", vmin=-1.0, vmax=1.0)
        axes[row, 0].set_title(f"Input MRI ({pids_b[row]})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(g_img, cmap="gray", vmin=-1.0, vmax=1.0)
        axes[row, 1].set_title("Ground-Truth CT")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(c_img, cmap="gray", vmin=-1.0, vmax=1.0)
        axes[row, 2].set_title("Stage-1 Coarse CT")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(s_img, cmap="gray", vmin=-1.0, vmax=1.0)
        axes[row, 3].set_title("Stage-2 CCR-Diffusion CT")
        axes[row, 3].axis("off")

        im_e = axes[row, 4].imshow(diff, cmap="hot", vmin=0.0, vmax=0.5)
        axes[row, 4].set_title("Masked Absolute Error")
        axes[row, 4].axis("off")
        fig.colorbar(im_e, ax=axes[row, 4], fraction=0.046, pad=0.04)

    plt.suptitle(f"CCR-Diffusion Qualitative Synthesis Grid — Epoch {epoch}", fontsize=14, y=0.98)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"ccr_diffusion_grid_epoch_{epoch:03d}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"[OK] Visual grid saved to: {save_path}")

print("[OK] Automated qualitative visualization module initialized.")''')

# ==============================================================================
# CELL 11: STAGE 2 — CCR-DIFFUSION TRAINING ENGINE
# ==============================================================================
add_code('''# =====================================================================
# STEP 5 & 6: STAGE-2 CCR-DIFFUSION TRAINING ENGINE
# =====================================================================

STAGE2_EPOCHS = 60
STAGE2_LR     = 1e-4
STAGE2_WD     = 1e-4
VAL_FREQ      = 5

# Select Primary Stage-1 Anchor Model (Fold 1 or Ensemble Anchor)
stage1_anchor = COARSE_MODELS[1].to(device)
stage1_anchor.eval()
for p in stage1_anchor.parameters():
    p.requires_grad = False

diff_optimizer = torch.optim.AdamW(ccr_diffusion_model.parameters(), lr=STAGE2_LR, weight_decay=STAGE2_WD)
diff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(diff_optimizer, T_max=STAGE2_EPOCHS, eta_min=1e-6)
diff_scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

best_val_score = float("inf")
best_epoch     = 0
training_history = []
validation_history = []

BEST_CKPT_PATH   = os.path.join(DIR_CHECKPOINTS, "best_checkpoint.pth")
LATEST_CKPT_PATH = os.path.join(DIR_CHECKPOINTS, "latest_checkpoint.pth")

print("=" * 80)
print(f"STARTING STAGE-2 CCR-DIFFUSION TRAINING ({STAGE2_EPOCHS} EPOCHS)")
print(f"  Conditioning Path   : Multi-scale Stage-1 ResUNet encoder feature modulation (FiLM)")
print(f"  Residual Scale s    : {RESIDUAL_SCALE:.4f}")
print(f"  Loss Formulation    : Masked Noise MSE + Aux Masked SSIM (t <= 250) | CT L1/MSE zeroed")
print(f"  Composite Metric    : mean_patient_MAE + 0.25 * (1 - mean_patient_SSIM)")
print(f"  Visual Grid Cadence : Every 10 epochs")
print("=" * 80)

start_train_time = time.time()

for epoch in range(1, STAGE2_EPOCHS + 1):
    ccr_diffusion_model.train()
    running_total_loss = 0.0
    running_noise_loss = 0.0
    running_ssim_loss  = 0.0

    pbar = tqdm(train_loader_stage2, desc=f"Epoch [{epoch:02d}/{STAGE2_EPOCHS}] Train CCR-Diffusion")
    for batch in pbar:
        mri    = batch["mri"].to(device)
        ct     = batch["ct"].to(device)
        coarse = batch["coarse"].to(device)
        mask   = batch["mask"].to(device)
        z_0    = batch["z_0"].to(device)
        b_sz   = z_0.size(0)

        # 1. Sample diffusion timesteps t in [0, 1000]
        t = torch.randint(0, sched.timesteps, (b_sz,), device=device).long()
        z_t, noise = sched.q_sample(z_0, t)

        # 2. Extract Stage-1 encoder features
        with torch.no_grad():
            cond_feats = stage1_anchor.extract_encoder_features(mri)

        diff_optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            # Denoiser input: [z_t, mask] (2 channels)
            model_in = torch.cat([z_t, mask], dim=1)
            pred_noise = ccr_diffusion_model(model_in, t, cond_feats)

            # Compute hybrid loss (Noise MSE + Aux SSIM for t <= 250)
            total_loss, l_noise, l_ssim = ccr_loss_engine(
                pred_noise, noise, mask, t, z_t, coarse, ct, RESIDUAL_SCALE, sched
            )

        diff_scaler.scale(total_loss).backward()
        diff_scaler.unscale_(diff_optimizer)
        torch.nn.utils.clip_grad_norm_(ccr_diffusion_model.parameters(), max_norm=1.0)
        diff_scaler.step(diff_optimizer)
        diff_scaler.update()

        running_total_loss += total_loss.item()
        running_noise_loss += l_noise.item()
        running_ssim_loss  += l_ssim.item() if isinstance(l_ssim, torch.Tensor) else l_ssim
        pbar.set_postfix({"Noise_MSE": f"{l_noise.item():.4f}", "Total": f"{total_loss.item():.4f}"})

    diff_scheduler.step()
    ep_total_loss = running_total_loss / len(train_loader_stage2)
    ep_noise_loss = running_noise_loss / len(train_loader_stage2)
    ep_ssim_loss  = running_ssim_loss / len(train_loader_stage2)
    cur_lr        = diff_optimizer.param_groups[0]["lr"]

    training_history.append({
        "epoch": epoch,
        "lr": cur_lr,
        "total_loss": ep_total_loss,
        "noise_mse": ep_noise_loss,
        "aux_ssim": ep_ssim_loss
    })

    # Always save latest checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": ccr_diffusion_model.state_dict(),
        "optimizer_state_dict": diff_optimizer.state_dict(),
        "residual_scale": RESIDUAL_SCALE,
        "epoch_loss": ep_total_loss
    }, LATEST_CKPT_PATH)

    # Periodic weights saving every 10 epochs
    if epoch % 10 == 0:
        periodic_path = os.path.join(DIR_CHECKPOINTS, f"epoch_{epoch:03d}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": ccr_diffusion_model.state_dict(),
            "residual_scale": RESIDUAL_SCALE
        }, periodic_path)

    # Automated grid visualization every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        render_visualization_grid(ccr_diffusion_model, stage1_anchor, epoch)

    # Validation evaluation
    run_val = (epoch % VAL_FREQ == 0) or (epoch == 1) or (epoch == STAGE2_EPOCHS)
    if run_val:
        print(f"--> Running validation evaluation across 6 validation subjects...")
        score, p_mae, p_ssim, p_psnr, df_p = evaluate_patient_level_validation(
            ccr_diffusion_model, stage1_anchor, ddim_steps=50
        )
        
        saved_star = ""
        if score < best_val_score:
            best_val_score = score
            best_epoch     = epoch
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": ccr_diffusion_model.state_dict(),
                "optimizer_state_dict": diff_optimizer.state_dict(),
                "residual_scale": RESIDUAL_SCALE,
                "best_score": best_val_score,
                "patient_mae": p_mae,
                "patient_ssim": p_ssim,
                "patient_psnr": p_psnr
            }, BEST_CKPT_PATH)
            saved_star = " ⭐ [BEST CHECKPOINT SAVED]"

        print(f"Epoch [{epoch:02d}/{STAGE2_EPOCHS}] | Train Noise MSE: {ep_noise_loss:.4f} | "
              f"Val Score: {score:.4f} (MAE: {p_mae:.4f}, SSIM: {p_ssim:.4f}, PSNR: {p_psnr:.2f} dB){saved_star}")

        validation_history.append({
            "epoch": epoch,
            "composite_score": score,
            "mean_patient_mae": p_mae,
            "mean_patient_ssim": p_ssim,
            "mean_patient_psnr": p_psnr
        })
    else:
        print(f"Epoch [{epoch:02d}/{STAGE2_EPOCHS}] | Train Noise MSE: {ep_noise_loss:.4f} (LR: {cur_lr:.2e})")

print("=" * 80)
print(f"CCR-DIFFUSION TRAINING COMPLETED in {(time.time() - start_train_time) / 60.0:.2f} minutes.")
print(f"  Best Validation Epoch : {best_epoch}")
print(f"  Best Composite Score  : {best_val_score:.4f}")
print("=" * 80)

# Save history logs to CSV
pd.DataFrame(training_history).to_csv(os.path.join(DIR_METRICS, "training_history.csv"), index=False)
pd.DataFrame(validation_history).to_csv(os.path.join(DIR_METRICS, "validation_history.csv"), index=False)''')

# ==============================================================================
# CELL 12: FINAL EVALUATION & VALIDATION REPORT EXPORT
# ==============================================================================
add_code('''# =====================================================================
# STEP 6 (CONT.): FROZEN BEST-CHECKPOINT EVALUATION & SUMMARY REPORT
# =====================================================================

print("Loading best checkpoint for frozen validation evaluation...")
ckpt_data = torch.load(BEST_CKPT_PATH, map_location=device)
ccr_diffusion_model.load_state_dict(ckpt_data["model_state_dict"])
ccr_diffusion_model.eval()

score, p_mae, p_ssim, p_psnr, df_patients = evaluate_patient_level_validation(
    ccr_diffusion_model, stage1_anchor, ddim_steps=50
)

# Export validation patient metrics
val_metrics_csv = os.path.join(DIR_METRICS, "final_validation_patient_metrics.csv")
df_patients.to_csv(val_metrics_csv, index=False)

print("=" * 80)
print("CCR-DIFFUSION FINAL VALIDATION RESULTS (PATIENT-LEVEL MEAN ± STD):")
print("=" * 80)
print(f"  Masked MAE (norm)         : {df_patients['patient_mae'].mean():.4f} ± {df_patients['patient_mae'].std():.4f}")
print(f"  Masked SSIM               : {df_patients['patient_ssim'].mean():.4f} ± {df_patients['patient_ssim'].std():.4f}")
print(f"  Masked PSNR (dB)          : {df_patients['patient_psnr'].mean():.2f} ± {df_patients['patient_psnr'].std():.2f}")
print(f"  Validation Composite Score: {score:.4f} (mean_patient_MAE + 0.25 * (1 - mean_patient_SSIM))")
print("=" * 80)
print(df_patients.to_string(index=False))
print("=" * 80)
print(f"[OK] Full deliverable metrics saved to: {val_metrics_csv}")
print("Note: Held-out test set (6 subjects) strictly preserved and isolated.")''')

# Create notebook dictionary
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_notebook_path = "SynthRAD_CCR_Diffusion.ipynb"
with open(out_notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"[SUCCESS] Built complete Jupyter notebook: {out_notebook_path}")
print(f"Total cells created: {len(cells)}")
