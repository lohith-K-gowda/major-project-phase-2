# -*- coding: utf-8 -*-
"""
generate_notebook.py - Generates publication-ready Kaggle/Local Notebook for ResUNet Baseline
"""

import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [text if text.endswith("\n") else text + "\n"]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()]
    })

# Cell 0: Markdown Title & Description
add_md("""# 🧠 SynthRAD2023: Supervised ResUNet Baseline (Brain MRI → CT Synthesis)
## Controlled Scientific Baseline for Comparison against Proposed CDDPM (Epoch 73) and Pix2Pix
### Key Principles & Research Standards:
1. **Identical Dataset & Split:** Uses the exact 60-patient cohort (48 Train / 6 Val / 6 Test) locked to `SEED = 42`.
2. **Strict Test Isolation:** Test patients (`1BA054`, `1BA097`, `1BA172`, `1BA234`, `1BA256`, `1BA307`) remain 100% frozen and untouched during training and model selection.
3. **Pure Supervised ResUNet:** Residual U-Net architecture without diffusion embeddings, attention gates, or discriminator.
4. **Leakage-Free Preprocessing:** Anatomical body mask derived **strictly from MRI**; zero CT leakage into input features.
5. **Loss Function:** Masked $L_1$ + $\\lambda_{ssim} \\cdot L_{SSIM}$ inside the MRI body contour.""")

# Cell 1: Environment & Device Setup
add_code("""import os
import glob
import math
import time
import json
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
from scipy.spatial.distance import directed_hausdorff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("[OK] Deterministic seed locked (SEED=42).")""")

# Cell 2: Dataset Discovery & Patient Split
add_code("""# =====================================================================
# DATASET DISCOVERY & PATIENT SPLIT (HISTORICAL EPOCH-73 SEED 42 SPLIT)
# =====================================================================

base_dir = "/kaggle/input/datasets/lohithkgowda/synthrad2023-brain-2d-png/synthrad_png"
if not os.path.exists(base_dir):
    for c in glob.glob("/kaggle/input/**/synthrad_png", recursive=True):
        if os.path.exists(os.path.join(c, "ct")) and os.path.exists(os.path.join(c, "mri")):
            base_dir = c
            break

if not os.path.exists(base_dir):
    # Local fallback
    for c in ["./synthrad_png", "../synthrad_png", "synthrad_experiments/baselines/resunet/data"]:
        if os.path.exists(os.path.join(c, "ct")):
            base_dir = c
            break

print(f"Dataset root identified: {base_dir}")
ct_dir, mri_dir = os.path.join(base_dir, "ct"), os.path.join(base_dir, "mri")

paired = sorted(list(set(os.listdir(ct_dir)).intersection(set(os.listdir(mri_dir)))))
pat_map = {}
for f in paired:
    pat_map.setdefault(f.split("_")[0], []).append(f)

for p in pat_map:
    pat_map[p] = sorted(pat_map[p])

# Verified Historical Patient Split (Seed 42)
VERIFIED_SPLIT = {
    "train": [
        "1BA001", "1BA005", "1BA012", "1BA014", "1BA022", "1BA032", "1BA040", "1BA058",
        "1BA075", "1BA082", "1BA085", "1BA091", "1BA100", "1BA103", "1BA105", "1BA116",
        "1BA131", "1BA143", "1BA151", "1BA158", "1BA159", "1BA164", "1BA175", "1BA184",
        "1BA185", "1BA189", "1BA201", "1BA206", "1BA220", "1BA222", "1BA227", "1BA239",
        "1BA247", "1BA253", "1BA260", "1BA266", "1BA278", "1BA288", "1BA292", "1BA294",
        "1BA305", "1BA325", "1BA328", "1BA336", "1BA345", "1BA358", "1BA359", "1BA368"
    ],
    "val": ["1BA076", "1BA111", "1BA125", "1BA141", "1BA300", "1BA379"],
    "test": ["1BA054", "1BA097", "1BA172", "1BA234", "1BA256", "1BA307"]
}

def get_paths(plist):
    m, c = [], []
    for p in plist:
        if p in pat_map:
            for f in pat_map[p]:
                m.append(os.path.join(mri_dir, f))
                c.append(os.path.join(ct_dir, f))
    return m, c

tr_m, tr_c = get_paths(VERIFIED_SPLIT["train"])
va_m, va_c = get_paths(VERIFIED_SPLIT["val"])
te_m, te_c = get_paths(VERIFIED_SPLIT["test"])

print(f"Data Loaded: Train={len(tr_m)} slices ({len(VERIFIED_SPLIT['train'])} pts) | "
      f"Val={len(va_m)} slices ({len(VERIFIED_SPLIT['val'])} pts) | "
      f"Test={len(te_m)} slices ({len(VERIFIED_SPLIT['test'])} pts) [FROZEN]")""")

# Cell 3: Dataset Class & Preprocessing
add_code("""# =====================================================================
# DATA PREPROCESSING & DATASET CLASS
# =====================================================================

def extract_body_contour_mask(img_arr, threshold=15):
    \"\"\"Extracts clean solid head contour, zeroing exterior scanner air.\"\"\"
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

class SynthRADResUNetDataset(Dataset):
    \"\"\"
    Paired Supervised Dataset for ResUNet.
    Input: MRI [1, H, W] in [-1.0, 1.0]
    Target: CT [1, H, W] in [-1.0, 1.0]
    Mask: MRI-derived body mask (NO CT LEAKAGE)
    \"\"\"
    def __init__(self, mri_paths, ct_paths, image_size=128, is_train=False):
        self.mri_paths = mri_paths
        self.ct_paths  = ct_paths
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return len(self.mri_paths)

    def _process(self, mri_path, ct_path):
        mri_img = Image.open(mri_path).convert('L')
        ct_img  = Image.open(ct_path).convert('L')
        
        mri_arr = np.array(mri_img, dtype=np.float32)
        ct_arr  = np.array(ct_img,  dtype=np.float32)
        
        # Mask derived strictly from MRI
        mri_mask = extract_body_contour_mask(mri_arr)
        mri_arr  = mri_arr * mri_mask
        
        # Resize
        mri_res  = cv2.resize(mri_arr,  (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        ct_res   = cv2.resize(ct_arr,   (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(mri_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # Scale to [-1.0, 1.0]
        norm_m = (mri_res / 255.0) * 2.0 - 1.0
        norm_c = (ct_res  / 255.0) * 2.0 - 1.0
        
        # Zero background
        norm_m[mask_res < 0.5] = -1.0
        norm_c[mask_res < 0.5] = -1.0
        
        return (torch.from_numpy(norm_m).unsqueeze(0).float(),
                torch.from_numpy(norm_c).unsqueeze(0).float(),
                torch.from_numpy(mask_res).unsqueeze(0).float())

    def __getitem__(self, idx):
        mri, ct, mask = self._process(self.mri_paths[idx], self.ct_paths[idx])

        if self.is_train:
            if random.random() > 0.5:
                mri  = TF.hflip(mri)
                ct   = TF.hflip(ct)
                mask = TF.hflip(mask)
            angle = random.uniform(-8.0, 8.0)
            tx, ty = int(random.uniform(-4, 4)), int(random.uniform(-4, 4))
            scale = random.uniform(0.96, 1.04)
            mri  = TF.affine(mri,  angle=angle, translate=[tx, ty], scale=scale, shear=0, fill=-1.0)
            ct   = TF.affine(ct,   angle=angle, translate=[tx, ty], scale=scale, shear=0, fill=-1.0)
            mask = TF.affine(mask, angle=angle, translate=[tx, ty], scale=scale, shear=0, fill=0.0)

        patient_id = os.path.basename(self.mri_paths[idx]).split("_")[0]
        return {
            "mri": mri,
            "ct": ct,
            "mask": mask,
            "patient_id": patient_id,
            "slice_name": os.path.basename(self.mri_paths[idx])
        }

train_loader = DataLoader(SynthRADResUNetDataset(tr_m, tr_c, is_train=True),  batch_size=8, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(SynthRADResUNetDataset(va_m, va_c, is_train=False), batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(SynthRADResUNetDataset(te_m, te_c, is_train=False), batch_size=8, shuffle=False, num_workers=2, pin_memory=True)""")

# Cell 4: ResUNet Architecture
add_code("""# =====================================================================
# SUPERVISED RESUNET ARCHITECTURE
# =====================================================================

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        groups = num_groups if out_channels % num_groups == 0 else (4 if out_channels % 4 == 0 else 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1  = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2  = nn.LeakyReLU(0.2, inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act2(h + residual)

class ResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super().__init__()
        f = base_filters
        self.enc1 = ResBlock(in_channels, f)        # 64
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResBlock(f, f * 2)               # 128
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResBlock(f * 2, f * 4)           # 256
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResBlock(f * 4, f * 8)           # 512
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ResBlock(f * 8, f * 8),
            ResBlock(f * 8, f * 8)
        )

        self.up4 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec4 = ResBlock(f * 8 + f * 4, f * 4)   # 256
        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec3 = ResBlock(f * 4 + f * 2, f * 2)   # 128
        self.up2 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec2 = ResBlock(f * 2 + f, f)           # 64
        self.up1 = nn.ConvTranspose2d(f, f, 2, stride=2)
        self.dec1 = ResBlock(f + f, f)               # 64

        self.final_conv = nn.Conv2d(f, out_channels, 1)
        self.final_act = nn.Tanh()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final_act(self.final_conv(d1))

model = ResUNet(in_channels=1, out_channels=1, base_filters=64).to(device)
print(f"[OK] ResUNet Loaded. Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")""")

# Cell 5: Loss Function & Evaluation Metrics
add_code("""# =====================================================================
# LOSS FUNCTION & CLINICAL EVALUATION METRICS
# =====================================================================

HU_MIN, HU_MAX, HU_RANGE = -1000.0, 1500.0, 2500.0
BONE_THRESHOLD_HU = 300.0

def norm_to_hu(norm_arr):
    unit = np.clip((norm_arr + 1.0) / 2.0, 0.0, 1.0)
    return unit * HU_RANGE + HU_MIN

class SSIMLoss(nn.Module):
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
        mu1, mu2 = F.conv2d(img1, w, padding=pad), F.conv2d(img2, w, padding=pad)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, w, padding=pad) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, w, padding=pad) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, w, padding=pad) - mu1_mu2
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)
        if mask is not None:
            m = mask.to(dtype=img1.dtype, device=img1.device)
            return ((1.0 - ssim_map) * m).sum() / (m.sum() + 1e-8)
        return 1.0 - ssim_map.mean()

class ResUNetLoss(nn.Module):
    def __init__(self, lambda_ssim=0.10):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.ssim_fn = SSIMLoss()

    def forward(self, pred, target, mask=None):
        diff = torch.abs(pred - target)
        if mask is not None:
            l1 = (diff * mask).sum() / (mask.sum() + 1e-8)
            l_ssim = self.ssim_fn(pred, target, mask=mask)
        else:
            l1 = diff.mean()
            l_ssim = self.ssim_fn(pred, target)
        total = l1 + self.lambda_ssim * l_ssim
        return total, l1, l_ssim

def compute_metrics(p_np, g_np, m_np):
    pred_c = np.clip(p_np, -1.0, 1.0)
    gt_c   = np.clip(g_np, -1.0, 1.0)
    pu, gu = (pred_c + 1.0) / 2.0, (gt_c + 1.0) / 2.0
    mae = float(np.mean(np.abs(pu - gu)))
    psnr = float(compute_psnr(gu, pu, data_range=1.0))
    ssim_val, smap = compute_ssim(gu, pu, data_range=1.0, full=True)
    body = m_np > 0.5
    m_mae = float(np.mean(np.abs(pu[body] - gu[body]))) if body.sum() > 0 else mae
    m_ssim = float(np.mean(smap[body])) if body.sum() > 0 else float(ssim_val)
    p_hu, g_hu = norm_to_hu(pred_c), norm_to_hu(gt_c)
    hu_mae = float(np.mean(np.abs(p_hu[body] - g_hu[body]))) if body.sum() > 0 else float(np.mean(np.abs(p_hu - g_hu)))
    
    # Bone Dice (> 300 HU)
    rb, fb = (g_hu > BONE_THRESHOLD_HU) & body, (p_hu > BONE_THRESHOLD_HU) & body
    inter = np.sum(rb & fb)
    tot = np.sum(rb) + np.sum(fb)
    dice = float((2.0 * inter) / (tot + 1e-8)) if tot > 0 else 1.0
    return {"MAE": mae, "PSNR": psnr, "SSIM": float(ssim_val), "Masked_SSIM": m_ssim, "HU_MAE": hu_mae, "Bone_Dice": dice}

criterion = ResUNetLoss(lambda_ssim=0.10).to(device)
print("[OK] Loss and evaluation metrics initialized.")""")

# Cell 6: Training Loop
add_code("""# =====================================================================
# PRODUCTION RESUNET TRAINING LOOP (50-80 EPOCHS)
# =====================================================================

NUM_EPOCHS = 80
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

best_val_loss = float('inf')
best_epoch = 0

train_history = []
val_history = []

print("=" * 80)
print(f"STARTING RESUNET BASELINE TRAINING ({NUM_EPOCHS} EPOCHS | AdamW 1e-4 | Masked L1 + 0.1 SSIM)")
print("=" * 80)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    r_loss, r_l1, r_ssim = 0.0, 0.0, 0.0
    for batch in train_loader:
        mri, ct, mask = batch["mri"].to(device), batch["ct"].to(device), batch["mask"].to(device)
        optimizer.zero_grad()
        pred = model(mri)
        loss, l1, l_s = criterion(pred, ct, mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        r_loss += loss.item()
        r_l1 += l1.item()
        r_ssim += l_s.item()

    lr_scheduler.step()
    cur_lr = optimizer.param_groups[0]["lr"]

    # Validation (STRICTLY ON VAL SET)
    model.eval()
    v_loss = 0.0
    v_maes, v_psnrs, v_ssims, v_hu_maes, v_dices = [], [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            mri, ct, mask = batch["mri"].to(device), batch["ct"].to(device), batch["mask"].to(device)
            pred = model(mri)
            loss, _, _ = criterion(pred, ct, mask=mask)
            v_loss += loss.item()
            p_np, g_np, m_np = pred.squeeze(1).cpu().numpy(), ct.squeeze(1).cpu().numpy(), mask.squeeze(1).cpu().numpy()
            for i in range(mri.size(0)):
                res = compute_metrics(p_np[i], g_np[i], m_np[i])
                v_maes.append(res["MAE"])
                v_psnrs.append(res["PSNR"])
                v_ssims.append(res["SSIM"])
                v_hu_maes.append(res["HU_MAE"])
                v_dices.append(res["Bone_Dice"])

    ep_tr_loss = r_loss / len(train_loader)
    ep_va_loss = v_loss / len(val_loader)
    ep_va_ssim = np.mean(v_ssims)
    ep_va_psnr = np.mean(v_psnrs)
    ep_va_hu   = np.mean(v_hu_maes)
    ep_va_dice = np.mean(v_dices)

    train_history.append({"epoch": epoch, "loss": ep_tr_loss, "lr": cur_lr})
    val_history.append({"epoch": epoch, "val_loss": ep_va_loss, "SSIM": ep_va_ssim, "PSNR": ep_va_psnr, "HU_MAE": ep_va_hu, "Bone_Dice": ep_va_dice})

    saved_str = ""
    if ep_va_loss < best_val_loss:
        best_val_loss = ep_va_loss
        best_epoch = epoch
        torch.save(model.state_dict(), "best_resunet_baseline.pth")
        saved_str = " ⭐ [BEST SAVED]"

    if epoch % 5 == 0 or epoch == 1 or saved_str != "":
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] (LR: {cur_lr:.1e}) | Train Loss: {ep_tr_loss:.4f} | "
              f"Val Loss: {ep_va_loss:.4f} (SSIM: {ep_va_ssim:.4f}, PSNR: {ep_va_psnr:.2f} dB, HU-MAE: {ep_va_hu:.1f} HU){saved_str}")

torch.save(model.state_dict(), "final_resunet_baseline.pth")
pd.DataFrame(train_history).to_csv("resunet_training_history.csv", index=False)
pd.DataFrame(val_history).to_csv("resunet_validation_metrics.csv", index=False)
print(f"[OK] Training complete. Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")""")

# Cell 7: Frozen Test Set Evaluation
add_code("""# =====================================================================
# FROZEN TEST SET EVALUATION (HELD-OUT 6 TEST PATIENTS)
# =====================================================================

print("=" * 80)
print("EVALUATING FROZEN BEST RESUNET MODEL ON ISOLATED TEST PATIENTS")
print("=" * 80)

model.load_state_dict(torch.load("best_resunet_baseline.pth", map_location=device))
model.eval()

slice_records = []
patient_groups = {p: [] for p in VERIFIED_SPLIT["test"]}

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating Test Patients"):
        mri  = batch["mri"].to(device)
        ct   = batch["ct"].to(device)
        mask = batch["mask"].to(device)
        pids = batch["patient_id"]
        names = batch["slice_name"]

        pred = model(mri)
        pred = torch.where(mask > 0.5, pred, torch.full_like(pred, -1.0))

        p_np, g_np, m_np, i_np = pred.squeeze(1).cpu().numpy(), ct.squeeze(1).cpu().numpy(), mask.squeeze(1).cpu().numpy(), mri.squeeze(1).cpu().numpy()

        for idx in range(mri.size(0)):
            pid, sname = pids[idx], names[idx]
            res = compute_metrics(p_np[idx], g_np[idx], m_np[idx])
            res["patient_id"] = pid
            res["slice_name"] = sname
            slice_records.append(res)
            patient_groups[pid].append((res["HU_MAE"], i_np[idx], g_np[idx], p_np[idx], sname))

df_slices = pd.DataFrame(slice_records)
df_slices.to_csv("resunet_test_slice_metrics.csv", index=False)

# Patient-wise Summary
pat_summary = []
for pid in VERIFIED_SPLIT["test"]:
    pdf = df_slices[df_slices["patient_id"] == pid]
    pat_summary.append({
        "Patient ID": pid,
        "Slices": len(pdf),
        "SSIM": f"{pdf['SSIM'].mean():.4f} ± {pdf['SSIM'].std():.4f}",
        "PSNR (dB)": f"{pdf['PSNR'].mean():.2f} ± {pdf['PSNR'].std():.2f}",
        "HU-MAE (HU)": f"{pdf['HU_MAE'].mean():.1f} ± {pdf['HU_MAE'].std():.1f}",
        "Bone Dice": f"{pdf['Bone_Dice'].mean():.4f} ± {pdf['Bone_Dice'].std():.4f}"
    })
df_patient = pd.DataFrame(pat_summary)
df_patient.to_csv("resunet_test_patient_metrics.csv", index=False)

# Aggregate Test Table
agg_report = {
    "Metric": ["SSIM (↑)", "PSNR (dB) (↑)", "MAE (↓)", "HU-MAE (HU) (↓)", "Bone Dice (↑)"],
    "ResUNet Baseline": [
        f"{df_slices['SSIM'].mean():.4f} ± {df_slices['SSIM'].std():.4f}",
        f"{df_slices['PSNR'].mean():.2f} ± {df_slices['PSNR'].std():.2f} dB",
        f"{df_slices['MAE'].mean():.4f} ± {df_slices['MAE'].std():.4f}",
        f"{df_slices['HU_MAE'].mean():.1f} ± {df_slices['HU_MAE'].std():.1f} HU",
        f"{df_slices['Bone_Dice'].mean():.4f} ± {df_slices['Bone_Dice'].std():.4f}"
    ]
}
df_agg = pd.DataFrame(agg_report)
df_agg.to_csv("resunet_test_summary_metrics.csv", index=False)

print("\n" + "=" * 80)
print("     FINAL AGGREGATE TEST METRICS (RESUNET BASELINE)      ")
print("=" * 80)
print(df_agg.to_string(index=False))
print("=" * 80)

print("\n" + "=" * 80)
print("     PATIENT-WISE TEST BREAKDOWN                         ")
print("=" * 80)
print(df_patient.to_string(index=False))
print("=" * 80)""")

# Cell 8: Publication-Quality Visualizations
add_code("""# =====================================================================
# PUBLICATION FIGURES: CONVERGENCE & QUALITATIVE COMPARISON
# =====================================================================

# 1. Loss & Validation Metrics
df_tr = pd.DataFrame(train_history)
df_va = pd.DataFrame(val_history)

plt.figure(figsize=(7, 4.5))
plt.plot(df_tr["epoch"], df_tr["loss"], label="Train Loss", lw=2, color="#1f77b4")
plt.plot(df_va["epoch"], df_va["val_loss"], label="Val Loss", lw=2, color="#ff7f0e")
plt.title("ResUNet Baseline: Training and Validation Convergence", fontweight="bold")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True, ls="--", alpha=0.5); plt.legend()
plt.tight_layout()
plt.savefig("FIG_RESUNET_TRAINING_LOSS.png", dpi=300)
plt.show()

# 2. Qualitative 4-Panel Grid: MRI | Ground Truth CT | ResUNet CT | Error Map
test_pts = VERIFIED_SPLIT["test"]
fig, axes = plt.subplots(len(test_pts), 4, figsize=(14, 3.2 * len(test_pts)))

for row, pid in enumerate(test_pts):
    slices = sorted(patient_groups[pid], key=lambda x: x[0])
    med_slice = slices[len(slices) // 2]
    med_mae, mri_img, gt_img, syn_img, sname = med_slice
    
    m_show = np.clip((mri_img + 1.0) / 2.0, 0, 1)
    g_show = np.clip((gt_img  + 1.0) / 2.0, 0, 1)
    s_show = np.clip((syn_img + 1.0) / 2.0, 0, 1)
    err = np.abs(s_show - g_show)
    
    axes[row, 0].imshow(m_show, cmap="gray")
    axes[row, 0].set_title(f"Patient {pid} (MRI)\n{sname}", fontsize=10); axes[row, 0].axis("off")
    
    axes[row, 1].imshow(g_show, cmap="gray")
    axes[row, 1].set_title("Ground Truth CT", fontsize=10); axes[row, 1].axis("off")
    
    axes[row, 2].imshow(s_show, cmap="gray")
    axes[row, 2].set_title(f"ResUNet CT\nHU-MAE: {med_mae:.1f} HU", fontsize=10, fontweight="bold", color="navy"); axes[row, 2].axis("off")
    
    im_err = axes[row, 3].imshow(err, cmap="inferno", vmin=0, vmax=0.30)
    axes[row, 3].set_title("Absolute Error Map", fontsize=10); axes[row, 3].axis("off")

plt.tight_layout()
cbar = fig.colorbar(im_err, ax=axes.ravel().tolist(), shrink=0.4, pad=0.02)
cbar.set_label("|Synthetic CT - Ground Truth|", fontsize=10)
plt.savefig("FIG_RESUNET_QUALITATIVE_RESULTS.png", dpi=300, bbox_inches="tight")
plt.show()
print("[OK] Publication figures saved: FIG_RESUNET_TRAINING_LOSS.png, FIG_RESUNET_QUALITATIVE_RESULTS.png")""")

# Cell 9: Comparative Analysis against Epoch 73 CDDPM
add_code("""# =====================================================================
# SCIENTIFIC COMPARISON: RESUNET BASELINE vs. EPOCH-73 CDDPM PROPOSED
# =====================================================================

comparison_data = {
    "Model": ["ResUNet (Baseline)", "CDDPM Epoch 73 (Proposed)"],
    "Type": ["Supervised Residual CNN", "Conditional Diffusion (DDPM)"],
    "Inference Mechanism": ["Single Forward Pass", "Iterative Reverse Diffusion (DDIM)"],
    "Parameters": ["18.5M", "33.7M"],
    "Test SSIM (↑)": [f"{df_slices['SSIM'].mean():.4f}", "0.8963 ± 0.0319"],
    "Test PSNR (dB) (↑)": [f"{df_slices['PSNR'].mean():.2f}", "24.35 ± 2.38"],
    "Test HU-MAE (HU) (↓)": [f"{df_slices['HU_MAE'].mean():.1f}", "57.4 ± 18.8"],
    "Test Bone Dice (↑)": [f"{df_slices['Bone_Dice'].mean():.4f}", "0.7732 ± 0.0696"]
}

df_comp = pd.DataFrame(comparison_data)
df_comp.to_csv("comparison_resunet_vs_cddpm_epoch73.csv", index=False)
print("=" * 80)
print("         MODEL BENCHMARK COMPARISON (SYNTHRAD2023)        ")
print("=" * 80)
print(df_comp.to_string(index=False))
print("=" * 80)""")

nb_content = {
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
    "nbformat_minor": 5
}

out_path = "synthrad_experiments/baselines/resunet/resunet_baseline_synthrad.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb_content, f, indent=1)

print(f"Notebook generated successfully at: {out_path} with {len(cells)} cells.")
