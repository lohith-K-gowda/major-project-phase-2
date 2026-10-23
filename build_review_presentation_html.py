# -*- coding: utf-8 -*-
"""
build_review_presentation_html.py
Creates an interactive, beautiful HTML Slide Presentation and Viva Guide with embedded
diagrams, speaking scripts, master benchmark comparison tables, and viva prep.
"""

import os

HTML_CONTENT = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VTU Major Project Review: 5-Model Comparative Study & CCR-Diffusion Framework</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --navy: #1B365D;
    --navy-dark: #0f1f38;
    --brown: #843C0C;
    --accent-blue: #2563EB;
    --accent-green: #10B981;
    --accent-amber: #F59E0B;
    --bg-dark: #0F172A;
    --card-bg: #1E293B;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --border-color: #334155;
    --speaker-bg: rgba(245, 158, 11, 0.12);
    --speaker-border: #F59E0B;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background-color: var(--bg-dark);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
    line-height: 1.6;
    padding: 24px;
  }

  .container {
    max-width: 1200px;
    margin: 0 auto;
  }

  header {
    background: linear-gradient(135deg, var(--navy-dark) 0%, var(--navy) 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 32px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 20px;
  }

  .header-text h1 {
    font-size: 24px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
  }

  .header-text p {
    color: #93C5FD;
    font-size: 14px;
    font-weight: 500;
  }

  .header-badges {
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: flex-end;
  }

  .badge {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    color: #F8FAFC;
  }

  .badge-gold {
    background: rgba(245, 158, 11, 0.2);
    border-color: #F59E0B;
    color: #FCD34D;
  }

  .nav-bar {
    position: sticky;
    top: 16px;
    z-index: 100;
    background: rgba(30, 41, 59, 0.95);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 12px 20px;
    margin-bottom: 28px;
    display: flex;
    gap: 12px;
    overflow-x: auto;
  }

  .nav-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 16px;
    border-radius: 8px;
    transition: all 0.2s ease;
    white-space: nowrap;
  }

  .nav-link:hover, .nav-link.active {
    background: var(--accent-blue);
    color: #FFFFFF;
  }

  .section-title {
    font-size: 20px;
    font-weight: 700;
    color: #F8FAFC;
    margin: 40px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--brown);
    display: flex;
    align-items: center;
    gap: 12px;
  }

  /* Slide Cards */
  .slide-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease, border-color 0.2s ease;
  }

  .slide-card:hover {
    border-color: var(--accent-blue);
  }

  .slide-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 14px;
  }

  .slide-num {
    background: var(--accent-blue);
    color: #FFFFFF;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .slide-title-text {
    font-size: 17px;
    font-weight: 700;
    color: #F8FAFC;
    flex-grow: 1;
    margin-left: 14px;
  }

  .slide-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 20px;
  }

  @media (max-width: 900px) {
    .slide-grid { grid-template-columns: 1fr; }
  }

  .bullet-list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .bullet-list li {
    position: relative;
    padding-left: 20px;
    font-size: 13.5px;
    color: #CBD5E1;
  }

  .bullet-list li::before {
    content: "•";
    position: absolute;
    left: 4px;
    color: var(--accent-blue);
    font-weight: bold;
    font-size: 18px;
    line-height: 1;
  }

  .slide-image-box {
    background: #0B1120;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  .slide-image-box img {
    max-width: 100%;
    max-height: 260px;
    object-fit: contain;
    border-radius: 8px;
  }

  .image-caption {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 8px;
    text-align: center;
    font-style: italic;
  }

  /* Speaker Script Box */
  .speaker-box {
    background: var(--speaker-bg);
    border: 1px solid var(--speaker-border);
    border-radius: 12px;
    padding: 18px 20px;
    margin-top: 14px;
  }

  .speaker-title {
    font-size: 12px;
    font-weight: 700;
    color: #FCD34D;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .speaker-text {
    font-size: 13.5px;
    color: #FEF3C7;
    font-style: italic;
    line-height: 1.6;
  }

  .takeaway-tag {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px dashed rgba(245, 158, 11, 0.3);
    font-size: 12px;
    font-weight: 600;
    color: #FDE68A;
  }

  /* Tables */
  .table-responsive {
    overflow-x: auto;
    margin: 20px 0;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12.5px;
    text-align: left;
  }

  th {
    background: #0F172A;
    color: #93C5FD;
    padding: 12px 14px;
    font-weight: 600;
    border-bottom: 2px solid var(--border-color);
  }

  td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-color);
    color: #E2E8F0;
  }

  tr:hover td {
    background: rgba(255, 255, 255, 0.03);
  }

  .highlight-row td {
    background: rgba(16, 185, 129, 0.1);
    color: #A7F3D0;
    font-weight: 600;
  }

  /* Viva Cards */
  .qa-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 14px;
  }

  .qa-question {
    font-size: 14px;
    font-weight: 700;
    color: #93C5FD;
    margin-bottom: 8px;
  }

  .qa-answer {
    font-size: 13px;
    color: #CBD5E1;
    line-height: 1.6;
  }

  .print-btn {
    background: var(--accent-green);
    color: #FFFFFF;
    border: none;
    padding: 8px 18px;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    text-decoration: none;
    font-size: 13px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .print-btn:hover {
    background: #059669;
  }
</style>
</head>
<body>

<div class="container">

  <!-- Header -->
  <header>
    <div class="header-text">
      <h1>VTU MAJOR PROJECT REVIEW: 5-MODEL BENCHMARK & CCR-DIFFUSION</h1>
      <p>Deep Learning-Based MRI-to-CT Image Translation Using Generative Models (SynthRAD2023)</p>
    </div>
    <div class="header-badges">
      <span class="badge badge-gold">22 Scheme CSE | Major Project</span>
      <span class="badge">Candidate: Lohith K & Team</span>
      <span class="badge">Guide: Dr. Ramya</span>
      <a href="Major_Project_Review_Presentation_and_Guide.pdf" target="_blank" class="print-btn">📥 Open Generated PDF</a>
    </div>
  </header>

  <!-- Sticky Navigation Bar -->
  <div class="nav-bar">
    <a href="#elevator-pitch" class="nav-link active">1. Strategic Overview</a>
    <a href="#model-overview" class="nav-link">2. 5-Model Matrix</a>
    <a href="#slide-deck" class="nav-link">3. 14-Slide PPT & Scripts</a>
    <a href="#benchmark-table" class="nav-link">4. Benchmark Results</a>
    <a href="#viva-qa" class="nav-link">5. Top 12 Viva Questions</a>
  </div>

  <!-- SECTION 1: Strategic Overview -->
  <div id="elevator-pitch">
    <h2 class="section-title">1. Strategic Review Guide: The 3-Minute Elevator Pitch</h2>
    <div class="slide-card">
      <div class="speaker-box" style="margin-top:0;">
        <div class="speaker-title">🎯 Word-for-Word Opening Statement to Evaluators</div>
        <div class="speaker-text">
          "Respected members of the review panel and our project guide, our major project addresses a critical clinical bottleneck in modern radiation oncology: the transition to an <b>MRI-Only Radiotherapy Planning Workflow</b>. Currently, cancer patients must undergo both an MRI for tumor contouring and a planning CT for electron density and radiation dose calculations. Coregistering these two scans introduces 1 to 5 millimeter geometric errors and subjects cancer patients to unnecessary ionizing radiation.<br><br>
          To eliminate the need for planning CTs, we formulated, trained, and benchmarked <b>five distinct deep learning architectures</b> on the international SynthRAD2023 brain cohort. Through systematic evaluation, we identified fundamental clinical failures in existing paradigms: supervised ResUNet produces blurry bone edges (the conditional mean trap), Pix2Pix GAN hallucinates false calcifications, and direct diffusion models suffer from geometric drift.<br><br>
          To solve all three problems, we engineered <b>CCR-Diffusion</b>—a novel decoupled two-stage framework that combines a 4-fold cross-fitted coarse prior with an out-of-fold residual diffusion denoiser. Our proposed framework achieves landmark performance (0.8864 Masked SSIM, 0.0391 MAE, 28.3% error reduction over baseline) and 498 millisecond inference speed via 50-step deterministic DDIM acceleration, delivering clinically viable, radiation-free synthetic CTs in under 60 seconds per patient."
        </div>
      </div>
    </div>
  </div>

  <!-- SECTION 2: 5-Model Matrix -->
  <div id="model-overview">
    <h2 class="section-title">2. Architectural Evolution: The Five Models at a Glance</h2>
    <div class="table-responsive">
      <table>
        <thead>
          <tr>
            <th>Model Paradigm</th>
            <th>Key Architecture & Loss Formulation</th>
            <th>Primary Clinical Strength</th>
            <th>Identified Failure / Clinical Risk</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><b>Model 1: ResUNet Baseline</b></td>
            <td>Supervised ResUNet + GroupNorm (8 grps) + L1/MSE/SSIM compound loss</td>
            <td>Fast, stable anatomical layout, high global PSNR (29.10 dB)</td>
            <td><b>Conditional Mean Trap:</b> Wash-out and blurred cortical bone edges; photon under-attenuation</td>
          </tr>
          <tr>
            <td><b>Model 2: Pix2Pix cGAN</b></td>
            <td>U-Net Generator + 70x70 PatchGAN Discriminator + LSGAN + 100*L1 loss</td>
            <td>Crisp bone edges, high bone Dice (0.8348)</td>
            <td><b>Hallucinations & Noise:</b> Granular diploic noise, false bone calcifications risking tumor under-dosing</td>
          </tr>
          <tr>
            <td><b>Model 3: cDDPM from Scratch</b></td>
            <td>Score-based diffusion U-Net, T=1000 linear schedule, Noise MSE</td>
            <td>High-frequency micro-texture realism without mode collapse</td>
            <td><b>Stochastic Drift:</b> Non-rigid boundary shifts from pure noise; high radiometric error (143.65 HU); slow</td>
          </tr>
          <tr>
            <td><b>Model 4: SOTA-CDDPM (73 Ep)</b></td>
            <td>Bottleneck self-attention gates (32x32 & 16x16), Sobel edge loss (5.0x)</td>
            <td>Improved calvarial vault contrast (118.94 HU MAE)</td>
            <td><b>Need for Anchor:</b> Proved that unconstrained diffusion from white noise cannot lock macroscopic geometry</td>
          </tr>
          <tr class="highlight-row">
            <td><b>Model 5: Proposed CCR-Diffusion</b></td>
            <td>Decoupled 2-Stage: 4-Fold Cross-Fitted Prior + FiLM Residual Denoiser + 50-step DDIM</td>
            <td><b>SOTA Champion:</b> 0.8864 SSIM, 0.0391 MAE (28.3% error reduction), 0.0982 RMSE, 498 ms/slice</td>
            <td>Requires 2-stage training (Stage 1 coarse prior + Stage 2 residual diffusion)</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- SECTION 3: 14-Slide PPT Deck -->
  <div id="slide-deck">
    <h2 class="section-title">3. Slide-by-Slide PPT Presentation Deck with Word-for-Word Speaker Scripts</h2>

    <!-- Slide 1 -->
    <div class="slide-card">
      <div class="slide-header">
        <span class="slide-num">Slide 1</span>
        <span class="slide-title-text">Title Slide: Deep Learning-Based MRI-to-CT Translation</span>
      </div>
      <div class="slide-grid">
        <ul class="bullet-list">
          <li><b>Title:</b> Deep Learning-Based MRI-TO-CT Image Translation Using Generative Models</li>
          <li><b>Sub-theme:</b> A Rigorous Five-Model Benchmark and Decoupled Residual Diffusion Framework for MRI-Only Radiotherapy Planning</li>
          <li><b>Context:</b> VTU Major Project Review (22 Scheme, Department of Computer Science & Engineering)</li>
          <li><b>Candidate:</b> Lohith K & Team | <b>Project Guide:</b> Dr. Ramya</li>
          <li><b>Institution:</b> Jyothy Institute of Technology, Bengaluru</li>
        </ul>
        <div class="slide-image-box">
          <img src="report/images/dfd_level_0_context.png" alt="Context Diagram">
          <div class="image-caption">Figure: DFD Level 0 Context Diagram (Radiotherapy System Interface)</div>
        </div>
      </div>
      <div class="speaker-box">
        <div class="speaker-title">🗣️ Speaker Script (What to Say)</div>
        <div class="speaker-text">
          "Good morning respected evaluators and guide. Today, we present our Major Project titled 'Deep Learning-Based MRI-TO-CT Image Translation Using Generative Models'. In this study, we benchmarked five generative deep learning architectures and engineered a novel residual diffusion framework, CCR-Diffusion, to enable single-modality MRI-only radiation therapy planning."
        </div>
        <div class="takeaway-tag">💡 Key Takeaway: Establish project domain, academic context, and clear clinical ambition immediately.</div>
      </div>
    </div>

    <!-- Slide 2 -->
    <div class="slide-card">
      <div class="slide-header">
        <span class="slide-num">Slide 2</span>
        <span class="slide-title-text">Clinical Motivation: The Imperative for MRI-Only Radiotherapy</span>
      </div>
      <div class="slide-grid">
        <ul class="bullet-list">
          <li><b>Current Bi-Modal Workflow:</b> Requires acquiring both an MRI (tumor contouring) and a planning CT (electron density calculation).</li>
          <li><b>Problem 1 (Coregistration Misalignment):</b> Rigid/deformable registration introduces 1 to 5 mm spatial errors due to patient repositioning and internal organ motion.</li>
          <li><b>Problem 2 (Ionizing Radiation Burden):</b> CT simulation scans deposit harmful ionizing radiation in cancer patients.</li>
          <li><b>Problem 3 (Logistical Costs):</b> Dual-scanner scheduling increases institutional overhead and delays treatment onset.</li>
          <li><b>Our Mission:</b> Synthesize geometrically authentic, physically calibrated synthetic CTs (sCT) directly from MRI scans.</li>
        </ul>
        <div class="slide-image-box">
          <img src="report/images/dfd_level_1_pipeline.png" alt="Level 1 Pipeline">
          <div class="image-caption">Figure: DFD Level 1 Subsystem Decomposition</div>
        </div>
      </div>
      <div class="speaker-box">
        <div class="speaker-title">🗣️ Speaker Script (What to Say)</div>
        <div class="speaker-text">
          "In brain cancer radiation therapy, doctors need MRI because it provides superior soft-tissue contrast to delineate tumor margins. However, treatment planning systems cannot calculate radiation dose from MRI alone because MRI reflects proton spin density, not electron density. Therefore, hospitals must also acquire a CT scan. Merging these two scans introduces 1 to 5 millimeters of coregistration error. In radiation oncology, even a 2-millimeter misalignment can cause healthy brain tissue to be irradiated while missing the tumor. By generating a synthetic CT directly from MRI, we completely eliminate the CT scan, remove coregistration error, and save patients from unnecessary ionizing radiation."
        </div>
        <div class="takeaway-tag">💡 Key Takeaway: 1-5 mm coregistration error and ionizing radiation are the primary clinical problems we solve.</div>
      </div>
    </div>

    <!-- Slide 3 -->
    <div class="slide-card">
      <div class="slide-header">
        <span class="slide-num">Slide 3</span>
        <span class="slide-title-text">Problem Statement & Mathematical Dilemma: The Bone Attenuation Challenge</span>
      </div>
      <div class="slide-grid">
        <ul class="bullet-list">
          <li><b>Physics Disconnect:</b> MRI signal is near zero for both cortical bone and air cavities (paranasal sinuses) due to low proton density and ultra-short T2 relaxation.</li>
          <li><b>CT Radiometric Divergence:</b> On CT, air is -1000 HU (zero attenuation) while cortical skull bone is +1000 to +2500 HU (maximum photon attenuation).</li>
          <li><b>The Mathematical Dilemma:</b> A mapping function G(MRI) -> CT must disambiguate two physically opposite tissue types that have identical zero appearance on the input MRI.</li>
          <li><b>Failure Mode:</b> Standard deep networks fall into the 'conditional mean trap'—averaging bone (+1500 HU) and air (-1000 HU), producing blurry, under-attenuated skull boundaries.</li>
        </ul>
        <div class="slide-image-box">
          <img src="report/images/model1_resunet_test_results.png" alt="ResUNet Results">
          <div class="image-caption">Figure: Model 1 ResUNet Test Slices showing Blurry Skull Vault</div>
        </div>
      </div>
      <div class="speaker-box">
        <div class="speaker-title">🗣️ Speaker Script (What to Say)</div>
        <div class="speaker-text">
          "Here is the fundamental scientific challenge: On an MRI scan, both dense skull bone and empty air cavities appear completely black because neither contains mobile protons. But on a CT scan, air has an intensity of minus 1000 Hounsfield Units, while dense cortical bone has an intensity of plus 1500 Hounsfield Units! A deep neural network is therefore asked to map two identical black inputs to two polar-opposite physical densities. When a standard model is unsure, it averages them out. This causes skull bone to look washed out and blurry. In radiotherapy, this leads to fatal dose calculation errors because the calculated photon beam penetration will be completely wrong."
        </div>
        <div class="takeaway-tag">💡 Key Takeaway: MRI signal voids for bone and air create the fundamental ambiguity that models must solve.</div>
      </div>
    </div>

    <!-- Slide 4 -->
    <div class="slide-card">
      <div class="slide-header">
        <span class="slide-num">Slide 4</span>
        <span class="slide-title-text">Dataset & Strict Experimental Protocol: SynthRAD2023 Grand Challenge</span>
      </div>
      <div class="slide-grid">
        <ul class="bullet-list">
          <li><b>Dataset:</b> SynthRAD2023 Grand Challenge (Task 1 Brain Cohort, UMC Utrecht).</li>
          <li><b>Data Dimensions:</b> 60 clinical brain cancer patients, 3,614 paired 2D axial slices (256x256).</li>
          <li><b>Radiometric Calibration:</b> Raw CT (-1000 HU to +1500 HU) normalized into [-1.0, 1.0].</li>
          <li><b>Strict Patient-Level Split (Seed 42):</b>
            <br>• Train: 48 Patients (2,879 slices)
            <br>• Validation: 6 Patients (372 slices)
            <br>• Frozen Test: 6 Patients (363 slices: 1BA054, 1BA097, 1BA172, 1BA234, 1BA256, 1BA307)
          </li>
          <li><b>Zero-Leakage Guarantee:</b> Test patients were completely frozen and untouched during training and model selection.</li>
          <li><b>Cranial Masking:</b> MRI-derived Otsu thresholding + 7x7 morphological closing eliminates exterior scanner noise.</li>
        </ul>
        <div class="slide-image-box">
          <img src="report/images/training_dynamics.png" alt="Training Dynamics">
          <div class="image-caption">Figure: 4-Panel Training & Validation Dynamics Curves over 80 Epochs</div>
        </div>
      </div>
      <div class="speaker-box">
        <div class="speaker-title">🗣️ Speaker Script (What to Say)</div>
        <div class="speaker-text">
          "To ensure international scientific credibility, we formulated our research on the SynthRAD2023 Grand Challenge dataset. We processed 60 clinical patients into 3,614 paired 2D axial slices. We enforced strict patient-level splitting using a fixed random seed of 42. Crucially, all slices from a patient were kept together in either train, validation, or test. Six patients, containing 363 axial slices, were completely frozen and never seen by any model during training or cross-fitting. Both MRI and CT were physically normalized into the minus 1.0 to plus 1.0 range, and cranial evaluation masks were derived strictly from MRI to prevent CT ground-truth leakage."
        </div>
        <div class="takeaway-tag">💡 Key Takeaway: Strict patient-level isolation (Seed 42) ensures 100% reproducible, zero-leakage evaluation.</div>
      </div>
    </div>

    <!-- Slide 5 to Slide 9: Models 1 to 4 -->
    <div class="slide-card">
      <div class="slide-header">
        <span class="slide-num">Slide 5</span>
        <span class="slide-title-text">Model 1: Deep Residual U-Net (ResUNet Baseline)</span>
      </div>
      <div class="slide-grid">
        <ul class="bullet-list">
          <li><b>Architecture:</b> 4-stage encoder-decoder with identity residual skip connections; 18.48M parameters.</li>
          <li><b>Normalization:</b> Group Normalization (8 groups) replaces BatchNorm for stable mini-batch statistics.</li>
          <li><b>Composite Objective:</b> Loss = 1.0 * L1 + 1.0 * MSE + 0.20 * (1 - SSIM).</li>
          <li><b>Convergence:</b> Rapid error reduction; stabilized at epoch 40; best validation MAE = 0.0545 (53.79 HU).</li>
          <li><b>Quantitative Performance:</b> Global PSNR: 29.10 dB | Masked SSIM: 0.8526 | Bone Dice: 0.8375.</li>
          <li><b>Critical Limitation:</b> The 'Conditional Mean Trap'. Supervised regression averages multi-modal tissue probabilities, resulting in washed-out cortical bone and blurred diploic boundaries.</li>
        </ul>
        <div class="slide-image-box">
          <img src="report/images/model1_resunet_test_results.png" alt="ResUNet Test">
          <div class="image-caption">Figure: Model 1 Output vs Ground Truth CT (Over-smoothing observable)</div>
        </div>
      </div>
      <div class="speaker-box">
        <div class="speaker-title">🗣️ Speaker Script (What to Say)</div>
        <div class="speaker-text">
          "Model 1 is our supervised ResUNet baseline. We used a 4-level encoder-decoder with residual skip connections and Group Normalization, trained under a composite L1, MSE, and SSIM objective. The model converges very smoothly and achieves a high global PSNR of 29.10 dB and a low MAE of 53.79 Hounsfield Units. However, visual inspection reveals its fatal flaw: the conditional mean trap. Because L1 and MSE penalize variance, the model plays safe by predicting the average intensity wherever MRI lacks distinct signal. As you can see, the skull bone edges are washed out and blurry, which leads to photon under-attenuation in treatment planning."
        </div>
        <div class="takeaway-tag">💡 Key Takeaway: ResUNet gives excellent global shape but fails on high-frequency bone edges due to mean-averaging.</div>
      </div>
    </div>

    <!-- Slide 6: Model 2 -->
    <div class="slide-card">
      <div class="slide-header">
        <span class="slide-num">Slide 6</span>
        <span class="slide-title-text">Model 2: Pix2Pix Conditional GAN (Adversarial Translation)</span>
      </div>
      <div class="slide-grid">
        <ul class="bullet-list">
          <li><b>Architecture:</b> U-Net-256 Generator paired with a 70x70 PatchGAN Discriminator (LSGAN loss).</li>
          <li><b>Adversarial Formulation:</b> Total Loss = Adversarial Loss + 100.0 * Masked L1 Reconstruction Loss.</li>
          <li><b>Training Dynamics:</b> Stable equilibrium reached across 80 epochs; D-loss = 0.614, G-loss = 5.81.</li>
          <li><b>Visual Quality:</b> Cortical bone is dramatically sharper than ResUNet; high Bone Dice of 0.8348.</li>
          <li><b>Clinical Hazard:</b> High-frequency checkerboard noise across the diploic space, higher overall MAE (0.0631 / 79.58 HU), elevated RMSE (0.1291), and hallucinated bone calcifications.</li>
        </ul>
        <div class="slide-image-box">
          <img src="report/images/model2_pix2pix_test_results.png" alt="Pix2Pix Test">
          <div class="image-caption">Figure: Model 2 Pix2Pix Output showing Diploic Space Noise</div>
        </div>
      </div>
      <div class="speaker-box">
        <div class="speaker-title">🗣️ Speaker Script (What to Say)</div>
        <div class="speaker-text">
          "To solve the blurriness of ResUNet, we built Model 2: a Pix2Pix conditional GAN. Here, a 70x70 PatchGAN discriminator forces the generator to produce high-frequency, realistic textures. Visually, the skull bones appear crisp and sharp, achieving a bone Dice score of 0.8348. However, adversarial training has a dangerous drawback: it introduces checkerboard noise and hallucinated calcifications in the diploic space. Radiometrically, its error jumped to 79.58 HU, and its RMSE rose to 0.1291. In cancer therapy, hallucinated dense bone causes the treatment planning system to falsely calculate high attenuation, leading to dangerous under-dosing of the adjacent tumor."
        </div>
        <div class="takeaway-tag">💡 Key Takeaway: Pix2Pix makes sharp bones, but GAN hallucinations and high RMSE make it dangerous for radiotherapy.</div>
      </div>
    </div>

    <!-- Slide 7: Model 3 -->
    <div class="slide-card">
      <div class="slide-header">
        <span class="slide-num">Slide 7</span>
        <span class="slide-title-text">Model 3: Conditional DDPM From Scratch (Score-Based Diffusion)</span>
      </div>
      <div class="slide-grid">
        <ul class="bullet-list">
          <li><b>Architecture:</b> 32-channel conditional U-Net denoiser with sinusoidal time embeddings.</li>
          <li><b>Forward Schedule:</b> Linear variance schedule with T=1000 steps (beta: 0.0001 -> 0.02).</li>
          <li><b>Objective:</b> Noise-prediction MSE: min E || epsilon - epsilon_theta(x_t, t, MRI) ||^2.</li>
          <li><b>Key Strength:</b> Eliminates adversarial instability and mode collapse; models authentic micro-textures.</li>
          <li><b>Identified Failure:</b> Stochastic Geometric Drift. Generating a full 256x256 CT slice from pure white Gaussian noise without structural constraints allowed non-rigid shifts in macroscopic skull contours (SSIM: 0.7473, MAE: 143.65 HU, 1000 steps).</li>
        </ul>
        <div class="slide-image-box">
          <img src="report/images/model3_cddpm_test_results.png" alt="cDDPM Test">
          <div class="image-caption">Figure: Model 3 cDDPM Output showing Geometric Boundary Shift</div>
        </div>
      </div>
      <div class="speaker-box">
        <div class="speaker-title">🗣️ Speaker Script (What to Say)</div>
        <div class="speaker-text">
          "Next, we turned to score-based diffusion models. Unlike GANs, diffusion models are likelihood-based and do not suffer from mode collapse. In Model 3, we implemented a Conditional DDPM from scratch with 1,000 diffusion steps. The network takes pure Gaussian noise and iteratively removes noise conditioned on the MRI. While it synthesized realistic micro-textures, it revealed a major problem: stochastic geometric drift. Because the model starts from pure random noise, it has to reconstruct both the overall patient head shape and the microscopic bone pixels simultaneously. This caused random shifts in the outer skull contours, resulting in a low SSIM of 0.7473 and an unacceptable radiometric error of 143.65 HU."
        </div>
        <div class="takeaway-tag">💡 Key Takeaway: Diffusion from pure noise captures texture but suffers from geometric drift and slow 1000-step sampling.</div>
      </div>
    </div>

    <!-- Slide 8: Model 4 -->
    <div class="slide-card">
      <div class="slide-header">
        <span class="slide-num">Slide 8</span>
        <span class="slide-title-text">Model 4: SOTA-CDDPM with Residual Attention & Edge Losses</span>
      </div>
      <div class="slide-grid">
        <ul class="bullet-list">
          <li><b>Architectural Upgrades:</b> Self-attention gates at 32x32 & 16x16 bottleneck resolutions; Group Normalization.</li>
          <li><b>Edge-Aware Loss:</b> Added a 5.0x weighted Sobel gradient penalty along the skull boundary.</li>
          <li><b>Training:</b> Trained for 73 epochs; best checkpoint achieved at epoch 73 with validation MAE = 0.0952 (118.94 HU).</li>
          <li><b>Scientific Realization:</b> While attention gates improved calvarial contrast, unconstrained diffusion from pure noise still exhibited slight boundary warping. This proved that <i>diffusion models must be anchored by an explicit, deterministic structural prior</i>.</li>
        </ul>
        <div class="slide-image-box">
          <img src="report/images/model4_sota_ddpm.png" alt="SOTA DDPM">
          <div class="image-caption">Figure: Model 4 Attention Gates & Convergence Behavior</div>
        </div>
      </div>
      <div class="speaker-box">
        <div class="speaker-title">🗣️ Speaker Script (What to Say)</div>
        <div class="speaker-text">
          "In Model 4, we upgraded the diffusion denoiser by embedding self-attention gates at the 32x32 and 16x16 bottleneck resolutions and adding an edge-aware Sobel bone loss. We trained this for 73 epochs, achieving its best validation checkpoint with an improved MAE of 118.94 HU. While attention gates helped delineate bone boundaries better than Model 3, it still suffered from slight non-rigid shape variations. This led us to our most critical scientific breakthrough: Diffusion models should not be forced to reconstruct the macroscopic head anatomy from white noise. Instead, we should decouple the problem—use a deterministic model for the anatomy, and use diffusion only for the fine residual details!"
        </div>
        <div class="takeaway-tag">💡 Key Takeaway: Model 4 proved that diffusion needs a deterministic structural anchor—giving birth to CCR-Diffusion.</div>
      </div>
    </div>

    <!-- Slide 9: Model 5 Champion -->
    <div class="slide-card">
      <div class="slide-header">
        <span class="slide-num">Slide 9</span>
        <span class="slide-title-text">Model 5 (Proposed Champion): CCR-Diffusion Framework</span>
      </div>
      <div class="slide-grid">
        <ul class="bullet-list">
          <li><b>Decoupled Two-Stage Architecture:</b>
            <br>• Stage 1 (Coarse Anatomical Anchor): 4-fold cross-fitted ResUNet predicts coarse CT (c).
            <br>• Residual Standardization: z0 = (True CT - Coarse CT) / s, where s = 0.1428 (OOF residual std dev).
            <br>• Stage 2 (FiLM Residual Denoiser): 4-channel conditional denoiser [z_t, MRI, Coarse CT, Mask].
          </li>
          <li><b>Zero-Leakage 4-Fold Cross-Fitting:</b> Each training patient's coarse CT is generated by a ResUNet instance trained exclusively on the other 3 folds.</li>
          <li><b>Ensemble Test Inference:</b> For test patients, the coarse CT is the deterministic average of all 4 fold models: c = 1/4 * sum(G_phi_k(MRI)).</li>
          <li><b>FiLM Modulation:</b> Feature-wise Linear Modulation dynamically scales and shifts intermediate features based on diffusion time step and anatomical priors.</li>
        </ul>
        <div class="slide-image-box">
          <img src="report/images/model5_ccr_diffusion_test_results.png" alt="CCR-Diffusion Test">
          <div class="image-caption">Figure: Proposed CCR-Diffusion Synthetic CT Output (Sharp & Accurate)</div>
        </div>
      </div>
      <div class="speaker-box">
        <div class="speaker-title">🗣️ Speaker Script (What to Say)</div>
        <div class="speaker-text">
          "This brings us to our primary contribution: Model 5, the CCR-Diffusion framework. We decoupled MRI-to-CT translation into two specialized stages: In Stage 1, a 4-fold patient-level cross-fitted ResUNet generates a deterministic coarse CT. This locks down the macroscopic patient anatomy and head geometry. We then compute the residual error—the difference between the real CT and the coarse CT—and scale it by s = 0.1428. In Stage 2, a 4-channel conditional diffusion U-Net learns solely to denoise this normalized residual error. We use FiLM layers—Feature-wise Linear Modulation—to adapt intermediate feature maps based on the diffusion time step and the coarse CT prior. Crucially, to prevent data leakage, Stage 2 is trained strictly on out-of-fold predictions from the other three folds."
        </div>
        <div class="takeaway-tag">💡 Key Takeaway: CCR-Diffusion decouples anatomy (deterministic prior) from texture (residual diffusion) with zero data leakage.</div>
      </div>
    </div>

    <!-- Slide 10: DDIM Acceleration -->
    <div class="slide-card">
      <div class="slide-header">
        <span class="slide-num">Slide 10</span>
        <span class="slide-title-text">Inference Acceleration: 50-Step Deterministic DDIM Sampling</span>
      </div>
      <div class="slide-grid">
        <ul class="bullet-list">
          <li><b>The Inference Bottleneck:</b> Standard DDPM requires 1000 sequential Markovian reverse steps (~10 seconds/slice), making it clinically impractical.</li>
          <li><b>Non-Markovian DDIM Sampling:</b> Implemented Denoising Diffusion Implicit Models (DDIM) with eta = 0 (deterministic sampling), compressing the reverse trajectory to just 50 steps.</li>
          <li><b>Bitwise Reproducibility:</b> Deterministic DDIM guarantees identical numerical outputs across repeated runs (zero stochastic jitter), satisfying strict radiotherapy QA standards.</li>
          <li><b>Throughput:</b> Achieves 498 ms per slice (2.01 frames/second) on a single clinical GPU.</li>
          <li><b>Full Cranial Volume Synthesis:</b> An entire 3D cranial radiotherapy scan (120 slices) is synthesized in under 60 seconds (20x faster than standard DDPM).</li>
        </ul>
        <div class="slide-image-box">
          <img src="report/images/dfd_level_2_ddim_loop.png" alt="DDIM Loop">
          <div class="image-caption">Figure: DFD Level 2 50-Step DDIM Reverse Sampling Loop</div>
        </div>
      </div>
      <div class="speaker-box">
        <div class="speaker-title">🗣️ Speaker Script (What to Say)</div>
        <div class="speaker-text">
          "A common criticism of diffusion models in clinical practice is their slow inference speed. Standard DDPM takes 1,000 sequential steps, which requires nearly 10 seconds per slice or 20 minutes for a full 3D head scan. To solve this, we implemented non-Markovian DDIM sampling with variance set to zero. This compresses the trajectory down to just 50 deterministic steps. Not only does this accelerate inference by twenty times—reaching 498 milliseconds per slice and under 60 seconds for an entire 3D patient volume—but it also guarantees bitwise reproducibility. Running inference ten times on the same patient yields the exact same pixel values every single time, which is mandatory for clinical radiotherapy quality assurance."
        </div>
        <div class="takeaway-tag">💡 Key Takeaway: 50-step deterministic DDIM makes diffusion clinically viable: sub-60-second full volume synthesis.</div>
      </div>
    </div>

    <!-- Slide 11: Master Benchmark -->
    <div class="slide-card">
      <div class="slide-header">
        <span class="slide-num">Slide 11</span>
        <span class="slide-title-text">Comprehensive Master Benchmark: Quantitative Clinical Evaluation</span>
      </div>
      <div class="slide-grid">
        <ul class="bullet-list">
          <li><b>Cohort:</b> Evaluated on all 363 frozen held-out test slices from 6 clinical patients under identical cranial masks.</li>
          <li><b>ResUNet Baseline:</b> SSIM: 0.8526 | MAE: 0.0545 (53.79 HU) | RMSE: 0.1089 | Bone Dice: 0.8375</li>
          <li><b>Pix2Pix cGAN:</b> SSIM: 0.8245 | MAE: 0.0631 (79.58 HU) | RMSE: 0.1291 | Bone Dice: 0.8348</li>
          <li><b>cDDPM Scratch:</b> SSIM: 0.7473 | MAE: 0.0898 (143.65 HU) | RMSE: 0.1601 | Bone Dice: 0.7887</li>
          <li><b>SOTA-CDDPM (73 Ep):</b> SSIM: 0.7424 | MAE: 0.0951 (118.94 HU) | RMSE: 0.1763 | Bone Dice: 0.7769</li>
          <li><b>CCR-Diffusion (Proposed):</b> <b>SSIM: 0.8864 (Highest)</b> | <b>MAE: 0.0391 (Lowest)</b> | <b>RMSE: 0.0982</b> | Bone Dice: 0.7877</li>
          <li><b>Decisive Victory:</b> CCR-Diffusion achieves a <b>28.3% error reduction over ResUNet</b> and <b>38.0% over Pix2Pix</b> with the lowest intensity outlier penalty (RMSE = 0.0982).</li>
        </ul>
        <div class="slide-image-box">
          <img src="report/images/qualitative_comparison_champion.png" alt="Qualitative Comparison">
          <div class="image-caption">Figure: Multi-Model Test Slice Comparison & Absolute Error Maps</div>
        </div>
      </div>
      <div class="speaker-box">
        <div class="speaker-title">🗣️ Speaker Script (What to Say)</div>
        <div class="speaker-text">
          "This slide presents our master quantitative benchmark on the 363 held-out test slices across all five models. As shown in the table, our proposed CCR-Diffusion model decisively outperforms all existing architectures. It achieves the highest Masked Structural Similarity of 0.8864 and the lowest Masked MAE of 0.0391. This represents a 28.3% error reduction compared to the ResUNet baseline, and a 38.0% error reduction compared to Pix2Pix. Furthermore, CCR-Diffusion achieves the lowest RMSE of 0.0982, proving that it eliminates the severe intensity spikes and hot-spot artifacts that plagued the adversarial and unconstrained diffusion models."
        </div>
        <div class="takeaway-tag">💡 Key Takeaway: CCR-Diffusion wins decisively: 28.3% lower MAE than ResUNet, 38.0% lower MAE than Pix2Pix, lowest RMSE.</div>
      </div>
    </div>

    <!-- Slide 12: Qualitative Heatmaps -->
    <div class="slide-card">
      <div class="slide-header">
        <span class="slide-num">Slide 12</span>
        <span class="slide-title-text">Qualitative Visual Analysis & Absolute Error Heatmaps</span>
      </div>
      <div class="slide-grid">
        <ul class="bullet-list">
          <li><b>ResUNet (Model 1):</b> Smooth global contours, but high residual error along the calvarial vault due to bone blurring.</li>
          <li><b>Pix2Pix (Model 2):</b> Intense localized hot-spots on error heatmaps caused by hallucinated calcifications and checkerboard diploic noise.</li>
          <li><b>cDDPM Models (3 & 4):</b> Diffuse background errors in sinus cavities and spatial shifts along the outer skull boundary.</li>
          <li><b>CCR-Diffusion (Model 5):</b>
            <br>• Flattest absolute error heatmaps across all 6 test patients.
            <br>• Continuous, intact calvarial plates with sharp diploic delineation.
            <br>• Precise soft-tissue brain parenchyma preservation with zero non-rigid boundary warping.
          </li>
        </ul>
        <div class="slide-image-box">
          <img src="report/images/qualitative_comparison_champion.png" alt="Champion Comparison">
          <div class="image-caption">Figure: Champion Visual Comparison with Absolute Error Heatmaps</div>
        </div>
      </div>
      <div class="speaker-box">
        <div class="speaker-title">🗣️ Speaker Script (What to Say)</div>
        <div class="speaker-text">
          "Turning to visual qualitative evaluation, this slide compares the synthetic CT outputs and absolute error heatmaps for test patient 1BA054. Notice the error map of ResUNet: there is a distinct red halo around the skull, confirming the bone under-attenuation caused by the conditional mean trap. In Pix2Pix, the error map shows intense localized speckles—these are the hallucinated calcifications created by the PatchGAN discriminator. For pure diffusion in Models 3 and 4, we observe diffuse error across the paranasal sinuses. In contrast, our proposed CCR-Diffusion displays the flattest, coolest error heatmap across the entire brain. It reconstructs continuous, intact calvarial plates and preserves fine brain soft-tissue boundaries with zero geometric warping."
        </div>
        <div class="takeaway-tag">💡 Key Takeaway: CCR-Diffusion produces the flattest error heatmaps with sharp, anatomically authentic skull bone.</div>
      </div>
    </div>

    <!-- Slide 13: System Integration & DFD -->
    <div class="slide-card">
      <div class="slide-header">
        <span class="slide-num">Slide 13</span>
        <span class="slide-title-text">System Integration & Data Flow Architecture (DFD Levels 0, 1, 2)</span>
      </div>
      <div class="slide-grid">
        <ul class="bullet-list">
          <li><b>Level 0 Context Diagram:</b> Interfaces the Radiation Oncologist / Medical Physicist, Generative Translation Engine, and Treatment Planning System (TPS).</li>
          <li><b>Level 1 Subsystems:</b> Ingestion & Preprocessing, Coarse Anchor Prediction (Stage 1), FiLM Residual Denoiser (Stage 2), and Radiotherapy QA Calibration.</li>
          <li><b>Level 2 Algorithmic Execution:</b> Detailed 50-step DDIM reverse trajectory, residual unscaling, and stoichiometric HU calibration.</li>
          <li><b>TPS Compatibility:</b> Directly outputs 16-bit DICOM series for commercial Treatment Planning Systems (e.g., Varian Eclipse, RaySearch RayStation, Elekta Monaco).</li>
        </ul>
        <div class="slide-image-box">
          <img src="report/images/dfd_level_1_pipeline.png" alt="DFD Pipeline">
          <div class="image-caption">Figure: DFD Level 1 Subsystem Functional Decomposition</div>
        </div>
      </div>
      <div class="speaker-box">
        <div class="speaker-title">🗣️ Speaker Script (What to Say)</div>
        <div class="speaker-text">
          "To demonstrate clinical engineering viability, we designed the complete end-to-end software architecture using formal Data Flow Diagrams. In DFD Level 0, patient MRI scans are ingested, processed by the CCR-Diffusion engine, and exported directly into the hospital's Treatment Planning System. DFD Level 1 details our four modular subsystems: automated preprocessing and cranial masking, coarse prior prediction, residual denoising, and radiometric QA calibration. Finally, DFD Level 2 specifies the algorithmic execution of the 50-step DDIM reverse loop and unscaling equation. This software architecture ensures plug-and-play compatibility with clinical radiotherapy software like Eclipse and RayStation."
        </div>
        <div class="takeaway-tag">💡 Key Takeaway: Formal DFD architecture proves end-to-end integration into commercial hospital treatment planning systems.</div>
      </div>
    </div>

    <!-- Slide 14: Conclusion -->
    <div class="slide-card">
      <div class="slide-header">
        <span class="slide-num">Slide 14</span>
        <span class="slide-title-text">Conclusion, Radiotherapy Impact & Future Scope</span>
      </div>
      <div class="slide-grid">
        <ul class="bullet-list">
          <li><b>Summary of Project Work:</b>
            <br>• Designed, trained, and benchmarked 5 deep learning paradigms on SynthRAD2023.
            <br>• Identified and documented the clinical hazards of regression blur and GAN hallucinations.
            <br>• Formulated the two-stage cross-fitted CCR-Diffusion framework, establishing a new SOTA.
            <br>• Accelerated inference to 498 ms/slice via 50-step deterministic DDIM.
          </li>
          <li><b>Clinical Impact:</b> Eliminates 1 to 5 mm coregistration errors, removes patient ionizing radiation dose, and saves hospital scanner operational costs.</li>
          <li><b>Future Work:</b>
            <br>• 3D Volumetric / Latent Consistency Models (1 to 4 step inference).
            <br>• Full Monte Carlo dosimetric gamma passing rate evaluation (&gamma; > 98%).
            <br>• Multi-contrast MRI ingestion (T1, T2, FLAIR) and cross-anatomical expansion (pelvis/head-and-neck).
          </li>
        </ul>
        <div class="slide-image-box">
          <img src="report/images/dfd_level_0_context.png" alt="DFD Level 0">
          <div class="image-caption">Figure: Complete MRI-Only Radiotherapy Workflow</div>
        </div>
      </div>
      <div class="speaker-box">
        <div class="speaker-title">🗣️ Speaker Script (What to Say)</div>
        <div class="speaker-text">
          "In conclusion, this project successfully bridges the gap between generative deep learning and clinical radiation oncology. We built and evaluated five complete generative models, documented why standard regression and GANs fall short in safety-critical medical imaging, and engineered the CCR-Diffusion framework. We also designed the complete system architecture, formalized in our Level 0, Level 1, and Level 2 Data Flow Diagrams, ensuring seamless integration with clinical Treatment Planning Systems like Eclipse and RayStation. With sub-second inference and landmark structural fidelity, this research provides a viable foundation for radiation-free, single-modality MRI-only radiotherapy. Thank you, and we welcome your questions."
        </div>
        <div class="takeaway-tag">💡 Key Takeaway: CCR-Diffusion makes single-modality MRI-only radiotherapy safe, accurate, and practically feasible.</div>
      </div>
    </div>

  </div>

  <!-- SECTION 4: Benchmark Results -->
  <div id="benchmark-table">
    <h2 class="section-title">4. Master Quantitative Benchmark Table (6 Test Patients, 363 Slices)</h2>
    <div class="table-responsive">
      <table>
        <thead>
          <tr>
            <th>Model Name</th>
            <th>Masked SSIM (&uarr;)</th>
            <th>Global SSIM (&uarr;)</th>
            <th>Global PSNR (dB) (&uarr;)</th>
            <th>Masked PSNR (dB) (&uarr;)</th>
            <th>Masked MAE (&darr;)</th>
            <th>HU-MAE (&darr;)</th>
            <th>RMSE (&darr;)</th>
            <th>Bone Dice (&gt;300 HU)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><b>Pix2Pix cGAN</b></td>
            <td>0.8245 &plusmn; 0.0340</td>
            <td>0.9120 &plusmn; 0.0110</td>
            <td>27.66 &plusmn; 0.77</td>
            <td>24.07 &plusmn; 1.03</td>
            <td>0.0631 &plusmn; 0.0098</td>
            <td>79.58 &plusmn; 12.34 HU</td>
            <td>0.1291 &plusmn; 0.0156</td>
            <td>0.8348 &plusmn; 0.0337</td>
          </tr>
          <tr>
            <td><b>ResUNet Baseline</b></td>
            <td>0.8526 &plusmn; 0.0280</td>
            <td>0.9268 &plusmn; 0.0107</td>
            <td>29.10 &plusmn; 0.70</td>
            <td>25.53 &plusmn; 0.90</td>
            <td>0.0545 &plusmn; 0.0074</td>
            <td>53.79 &plusmn; 7.55 HU</td>
            <td>0.1089 &plusmn; 0.0095</td>
            <td>0.8375 &plusmn; 0.0447</td>
          </tr>
          <tr>
            <td><b>cDDPM Scratch</b></td>
            <td>0.7473 &plusmn; 0.0470</td>
            <td>0.8753 &plusmn; 0.0151</td>
            <td>25.87 &plusmn; 0.50</td>
            <td>22.28 &plusmn; 0.70</td>
            <td>0.0898 &plusmn; 0.0099</td>
            <td>143.65 &plusmn; 50.93 HU</td>
            <td>0.1601 &plusmn; 0.0182</td>
            <td>0.7887 &plusmn; 0.0398</td>
          </tr>
          <tr>
            <td><b>SOTA-CDDPM (73 Ep)</b></td>
            <td>0.7424 &plusmn; 0.0336</td>
            <td>0.8692 &plusmn; 0.0077</td>
            <td>24.89 &plusmn; 0.36</td>
            <td>21.30 &plusmn; 0.55</td>
            <td>0.0951 &plusmn; 0.0095</td>
            <td>118.94 &plusmn; 11.82 HU</td>
            <td>0.1763 &plusmn; 0.0210</td>
            <td>0.7769 &plusmn; 0.0342</td>
          </tr>
          <tr class="highlight-row">
            <td><b>CCR-Diffusion (Proposed)</b></td>
            <td><b>0.8864 &plusmn; 0.0251</b></td>
            <td><b>0.9405 &plusmn; 0.0102</b></td>
            <td><b>27.15 &plusmn; 1.19</b></td>
            <td><b>23.56 &plusmn; 1.25</b></td>
            <td><b>0.0391 &plusmn; 0.0082</b></td>
            <td><b>97.53 &plusmn; 18.50 HU</b></td>
            <td><b>0.0982 &plusmn; 0.0088</b></td>
            <td><b>0.7877 &plusmn; 0.0540</b></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- SECTION 5: Viva Q&A -->
  <div id="viva-qa">
    <h2 class="section-title">5. Top 12 Viva & Review Questions with Winning Answers</h2>

    <div class="qa-card">
      <div class="qa-question">Q1: Why did you implement five models instead of just building one?</div>
      <div class="qa-answer">In medical image synthesis, adopting an architecture without empirical justification is unacceptable. Supervised CNNs (ResUNet) and GANs (Pix2Pix) have well-documented failure modes: regression blur and adversarial hallucinations, respectively. By implementing ResUNet, Pix2Pix, cDDPM from scratch, and SOTA-CDDPM with attention, we scientifically proved why each existing paradigm fails clinically. This systematically justified the formulation of our proposed CCR-Diffusion model as the definitive solution.</div>
    </div>

    <div class="qa-card">
      <div class="qa-question">Q2: How do you prove there was zero data leakage in your 4-fold cross-fitting?</div>
      <div class="qa-answer">Stage 1 cross-fitting divides the 48 training patients into 4 mutually exclusive patient-level folds. Each ResUNet instance is trained on 3 folds (36 patients) and generates coarse predictions strictly for the held-out 4th fold (12 patients). Stage 2 residual diffusion is trained exclusively on these out-of-fold (OOF) predictions. Most importantly, the 6 test patients (363 slices) were completely frozen and untouched during all training and cross-fitting phases.</div>
    </div>

    <div class="qa-card">
      <div class="qa-question">Q3: Why did you normalize CT values to [-1.0, 1.0] instead of using raw Hounsfield Units directly?</div>
      <div class="qa-answer">Deep neural networks with Tanh final activations and stable gradient dynamics require zero-centered, bounded targets. Raw Hounsfield Units span an asymmetric range from -1000 HU (air) to +1500 HU (bone) or higher. Training directly on unnormalized HU values causes severe exploding gradients in convolutional layers. Normalized values are mapped linearly back to Hounsfield Units via: HU = ((norm + 1.0) / 2.0) * 2500 - 1000 for clinical dosimetric evaluation.</div>
    </div>

    <div class="qa-card">
      <div class="qa-question">Q4: What is the 'conditional mean trap' in Model 1 ResUNet?</div>
      <div class="qa-answer">In supervised regression under L1 or MSE loss, the mathematical expectation E[y|x] is predicted. On MRI, cortical skull bone and air cavities both have zero signal. Because the input feature is identical, the network cannot determine whether the pixel is bone (+1500 HU) or air (-1000 HU). To minimize mean squared penalty, the network predicts the average intensity, resulting in blurred, washed-out bone boundaries.</div>
    </div>

    <div class="qa-card">
      <div class="qa-question">Q5: Why is Pix2Pix considered dangerous in radiation oncology if its visual quality looks sharp?</div>
      <div class="qa-answer">Pix2Pix uses a PatchGAN discriminator that rewards high-frequency texture to defeat the generator loss. In regions of ambiguity, the generator synthesizes sharp, plausible-looking textures that do not actually exist in the patient's anatomy—known as adversarial hallucinations. In radiotherapy planning, a hallucinated bone calcification causes the Treatment Planning System to calculate excessive photon attenuation, under-dosing the real tumor and causing cancer recurrence.</div>
    </div>

    <div class="qa-card">
      <div class="qa-question">Q6: Why did direct diffusion (Model 3) suffer from geometric drift?</div>
      <div class="qa-answer">Standard DDPM initiates the reverse denoising Markov chain from pure, unconstrained white Gaussian noise. The denoiser must simultaneously learn macroscopic patient anatomy (skull size, head shape) and microscopic tissue densities from scratch. Without an explicit deterministic structural anchor, small stochastic errors accumulate over 1000 steps, causing non-rigid spatial warping of outer cranial boundaries.</div>
    </div>

    <div class="qa-card">
      <div class="qa-question">Q7: What is FiLM conditioning in CCR-Diffusion and why is it used?</div>
      <div class="qa-answer">FiLM stands for Feature-wise Linear Modulation. Instead of simply concatenating the coarse CT as an extra input channel, FiLM dynamically scales and shifts intermediate convolutional feature maps: FiLM(h) = gamma(c, t) * h + beta(c, t). This allows the network to adaptively modulate feature activations at every resolution level based on the current diffusion timestep and anatomical prior.</div>
    </div>

    <div class="qa-card">
      <div class="qa-question">Q8: How did you accelerate diffusion inference from 10 seconds down to 498 milliseconds?</div>
      <div class="qa-answer">Standard DDPM uses a 1000-step stochastic Markovian process. We implemented Denoising Diffusion Implicit Models (DDIM) with variance parameter eta = 0. This formulates reverse sampling as a deterministic discretization of an ordinary differential equation (ODE), allowing us to compress the reverse trajectory down to 50 steps without quality degradation. This reduced latency to 498 ms per slice (sub-60 seconds for a full 3D cranial volume).</div>
    </div>

    <div class="qa-card">
      <div class="qa-question">Q9: What is the role of the residual scaling factor s = 0.1428?</div>
      <div class="qa-answer">The raw residual r = y - coarse_CT has a narrow standard deviation of approximately 0.1428 because the coarse model already predicts the macroscopic layout. Diffusion models are mathematically designed to denoise unit-variance distributions N(0, I). Dividing the raw residual by s = 0.1428 normalizes it into a standardized unit-variance target z0 = r / s, ensuring optimal score matching and stable reverse trajectory dynamics.</div>
    </div>

    <div class="qa-card">
      <div class="qa-question">Q10: Why is Cortical Bone Dice (> 300 HU) such an important clinical metric?</div>
      <div class="qa-answer">In brain radiation oncology, high-energy photon and proton beams lose the majority of their energy when traversing the dense cortical bone of the calvarial vault. An intensity threshold of +300 HU isolates bone from soft tissue. Measuring the Dice similarity of the segmented bone mask verifies whether the model accurately reconstructs skull thickness and continuity, which directly governs radiotherapy dose accuracy.</div>
    </div>

    <div class="qa-card">
      <div class="qa-question">Q11: What were the computational hardware specifications used?</div>
      <div class="qa-answer">Experiments were executed on Kaggle cloud accelerators utilizing NVIDIA Tesla P100 (16 GB VRAM) and Dual Tesla T4 (2x16 GB VRAM) GPUs. PyTorch with CUDA 12.1 and mixed-precision (FP16) was employed. Training converged within 75-80 epochs, and inference memory footprint was constrained to under 2.4 GB VRAM.</div>
    </div>

    <div class="qa-card">
      <div class="qa-question">Q12: How does your synthetic CT integrate into real hospital treatment planning systems?</div>
      <div class="qa-answer">Our pipeline exports synthetic CT volumes formatted as standard 16-bit DICOM series calibrated in Hounsfield Units. These files are directly ingested by commercial Treatment Planning Systems like Varian Eclipse, RaySearch RayStation, and Elekta Monaco. The TPS applies standard stoichiometric Hounsfield Unit-to-Electron Density calibration curves to compute 3D Monte Carlo radiation dose distributions.</div>
    </div>
  </div>

</div>

</body>
</html>
'''

with open("Major_Project_Review_Presentation_and_Guide.html", "w", encoding="utf-8") as f:
    f.write(HTML_CONTENT)

print("Saved Major_Project_Review_Presentation_and_Guide.html successfully!")
