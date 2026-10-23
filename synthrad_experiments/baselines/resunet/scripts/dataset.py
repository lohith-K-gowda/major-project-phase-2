# -*- coding: utf-8 -*-
"""
dataset.py - SynthRAD2023 Paired Brain MRI -> CT Dataset Pipeline for ResUNet
- Pure MRI-supervised conditioning (zero CT information leaked into inputs)
- Exact patient-level 80/10/10 split matching historical Epoch-73 CDDPM (Seed 42)
- Body contour extraction & physical [-1.0, 1.0] intensity normalization
- Synchronous spatial augmentations on training set
"""

import os
import glob
import json
import random
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

def extract_body_contour_mask(img_arr, threshold=15):
    """
    Extracts anatomical body contour mask for head/brain MRI,
    eliminating exterior scanner air noise while preserving brain and ventricles.
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

class SynthRADResUNetDataset(Dataset):
    """
    Supervised paired dataset for ResUNet MRI -> CT synthesis.
    Input: MRI [1, H, W] in [-1.0, 1.0]
    Target: CT [1, H, W] in [-1.0, 1.0]
    Mask: MRI-derived body mask [1, H, W] {0.0, 1.0}
    """
    def __init__(self, mri_paths, ct_paths, image_size=128, is_train=False):
        assert len(mri_paths) == len(ct_paths), f"Mismatch: {len(mri_paths)} MRI vs {len(ct_paths)} CT paths"
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
        
        # Body contour mask derived STRICTLY from MRI (no CT leakage)
        mri_mask = extract_body_contour_mask(mri_arr)
        mri_arr  = mri_arr * mri_mask
        
        # Resize to model dimension
        mri_res  = cv2.resize(mri_arr,  (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        ct_res   = cv2.resize(ct_arr,   (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(mri_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # Normalize to [-1.0, 1.0]
        mri_norm = (mri_res / 255.0) * 2.0 - 1.0
        ct_norm  = (ct_res  / 255.0) * 2.0 - 1.0
        
        # Enforce exterior scanner background strictly at -1.0
        mri_norm[mask_res < 0.5] = -1.0
        ct_norm[mask_res < 0.5]  = -1.0
        
        t_mri  = torch.from_numpy(mri_norm).unsqueeze(0).float()
        t_ct   = torch.from_numpy(ct_norm).unsqueeze(0).float()
        t_mask = torch.from_numpy(mask_res).unsqueeze(0).float()
        
        return t_mri, t_ct, t_mask

    def __getitem__(self, idx):
        mri, ct, mask = self._process_slice(self.mri_paths[idx], self.ct_paths[idx])
        
        if self.is_train:
            # Synchronous horizontal flip
            if random.random() > 0.5:
                mri  = TF.hflip(mri)
                ct   = TF.hflip(ct)
                mask = TF.hflip(mask)
                
            # Synchronous mild affine transformations
            angle = random.uniform(-8.0, 8.0)
            tx = int(random.uniform(-4, 4))
            ty = int(random.uniform(-4, 4))
            scale = random.uniform(0.96, 1.04)
            
            mri  = TF.affine(mri,  angle=angle, translate=[tx, ty], scale=scale, shear=0, fill=-1.0)
            ct   = TF.affine(ct,   angle=angle, translate=[tx, ty], scale=scale, shear=0, fill=-1.0)
            mask = TF.affine(mask, angle=angle, translate=[tx, ty], scale=scale, shear=0, fill=0.0)

        patient_id = os.path.basename(self.mri_paths[idx]).split("_")[0]
        slice_name = os.path.basename(self.mri_paths[idx])

        return {
            "mri": mri,
            "ct":  ct,
            "mask": mask,
            "patient_id": patient_id,
            "slice_name": slice_name,
            "mri_path": self.mri_paths[idx],
            "ct_path": self.ct_paths[idx]
        }

def find_dataset_root(candidate_roots=None):
    """
    Auto-detects dataset directory across Kaggle paths and local directories.
    """
    if candidate_roots is None:
        candidate_roots = [
            "/kaggle/input/datasets/lohithkgowda/synthrad2023-brain-2d-png/synthrad_png",
            "/kaggle/input/synthrad2023-brain-2d-png/synthrad_png",
            "./synthrad_png",
            "../synthrad_png"
        ]
        
    for path in candidate_roots:
        if os.path.exists(path) and os.path.isdir(os.path.join(path, "ct")) and os.path.isdir(os.path.join(path, "mri")):
            return path
            
    # Recursive search in /kaggle/input if available
    for match in glob.glob("/kaggle/input/**/synthrad_png", recursive=True):
        if os.path.isdir(os.path.join(match, "ct")) and os.path.isdir(os.path.join(match, "mri")):
            return match

    return None

def get_synthrad_loaders(base_dir, split_json_path=None, image_size=128, batch_size=8, seed=42, num_workers=2):
    """
    Constructs train, validation, and test loaders using the exact verified patient split.
    """
    ct_dir  = os.path.join(base_dir, "ct")
    mri_dir = os.path.join(base_dir, "mri")
    
    ct_files  = set(os.listdir(ct_dir))
    mri_files = set(os.listdir(mri_dir))
    paired    = sorted(list(ct_files.intersection(mri_files)))
    
    pat_map = {}
    for f in paired:
        p = f.split("_")[0]
        pat_map.setdefault(p, []).append(f)
        
    for p in pat_map:
        pat_map[p] = sorted(pat_map[p])
        
    # Load or generate patient split
    if split_json_path and os.path.exists(split_json_path):
        with open(split_json_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        train_p = sorted(split_data["train"])
        val_p   = sorted(split_data["val"])
        test_p  = sorted(split_data["test"])
    else:
        # Deterministic permutation using Seed 42
        patients = sorted(list(pat_map.keys()))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(patients).tolist()
        n_tr = int(round(len(patients) * 0.80))
        n_va = int(round(len(patients) * 0.10))
        train_p = sorted(perm[:n_tr])
        val_p   = sorted(perm[n_tr:n_tr + n_va])
        test_p  = sorted(perm[n_tr + n_va:])

    def collect(plist):
        m, c = [], []
        for p in plist:
            if p in pat_map:
                for f in pat_map[p]:
                    m.append(os.path.join(mri_dir, f))
                    c.append(os.path.join(ct_dir, f))
        return m, c

    tr_m, tr_c = collect(train_p)
    va_m, va_c = collect(val_p)
    te_m, te_c = collect(test_p)

    train_ds = SynthRADResUNetDataset(tr_m, tr_c, image_size=image_size, is_train=True)
    val_ds   = SynthRADResUNetDataset(va_m, va_c, image_size=image_size, is_train=False)
    test_ds  = SynthRADResUNetDataset(te_m, te_c, image_size=image_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    split_info = {
        "train_patients": train_p,
        "val_patients": val_p,
        "test_patients": test_p,
        "train_slices": len(tr_m),
        "val_slices": len(va_m),
        "test_slices": len(te_m)
    }

    return train_loader, val_loader, test_loader, split_info
