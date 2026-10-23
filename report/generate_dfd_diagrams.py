import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

def create_dfd_level_0(output_path="report/images/dfd_level_0_context.png"):
    # Wide canvas to give ample space for all text and arrows
    fig, ax = plt.subplots(figsize=(14.2, 4.6), dpi=300)
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 4.6)
    ax.axis('off')

    # Card background
    bg = FancyBboxPatch((0.15, 0.15), 13.9, 4.3, boxstyle="round,pad=0.1,rounding_size=0.18",
                         facecolor="#FAFAFA", edgecolor="#D6D6D6", linewidth=1.5, zorder=0)
    ax.add_patch(bg)

    # 1. Left Entity: Radiation Oncologist / Medical Physicist
    # x in [0.6, 3.2] (w=2.6)
    box_left = FancyBboxPatch((0.6, 1.1), 2.6, 2.1, boxstyle="round,pad=0.15,rounding_size=0.15",
                              facecolor="#EEF2FF", edgecolor="#3730A3", linewidth=2.2, zorder=2)
    ax.add_patch(box_left)
    ax.text(1.9, 2.15, "Radiation\nOncologist /\nMedical Physicist", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#1E1B4B", zorder=3)

    # 2. Center Process: 0.0 Generative MRI-to-CT Translation System
    # Center at (7.1, 2.15), w=3.0, h=2.6 -> x in [5.6, 8.6]
    oval_center = Ellipse((7.1, 2.15), width=3.0, height=2.6,
                          facecolor="#FFFBEB", edgecolor="#B45309", linewidth=2.4, zorder=2)
    ax.add_patch(oval_center)
    ax.text(7.1, 2.8, "0.0", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#92400E", zorder=3)
    ax.text(7.1, 2.1, "Generative\nMRI-to-CT Translation\nSystem", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#1C1917", zorder=3)
    ax.text(7.1, 1.4, "(CCR-Diffusion Engine)", ha="center", va="center",
            fontsize=9.5, fontstyle="italic", color="#78350F", zorder=3)

    # 3. Right Entity: Radiation Treatment Planning System (TPS)
    # x in [11.0, 13.6] (w=2.6)
    box_right = FancyBboxPatch((11.0, 1.1), 2.6, 2.1, boxstyle="round,pad=0.15,rounding_size=0.15",
                               facecolor="#F0FDF4", edgecolor="#166534", linewidth=2.2, zorder=2)
    ax.add_patch(box_right)
    ax.text(12.3, 2.15, "Radiation\nTreatment Planning\nSystem (TPS)", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#14532D", zorder=3)

    # Left to Center Gap: x in [3.2, 5.6] -> width = 2.4 units!
    # Arrow 1: User -> System (Upper Arrow, pointing right)
    arrow1 = FancyArrowPatch((3.35, 2.7), (5.45, 2.7),
                             arrowstyle='-|>', mutation_scale=18,
                             color="#2563EB", linewidth=2.4, zorder=4)
    ax.add_patch(arrow1)
    ax.text(4.4, 3.0, "Patient MRI Scans\n& Acquisition Specs", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color="#1D4ED8", zorder=5)

    # Arrow 2: System -> User (Lower Arrow, pointing left)
    arrow2 = FancyArrowPatch((5.45, 1.6), (3.35, 1.6),
                             arrowstyle='-|>', mutation_scale=18,
                             color="#BE185D", linewidth=2.4, zorder=4)
    ax.add_patch(arrow2)
    ax.text(4.4, 1.25, "Dose / Error Metrics\n& QA Reports", ha="center", va="top",
            fontsize=9.5, fontweight="bold", color="#9D174D", zorder=5)

    # Center to Right Gap: x in [8.6, 11.0] -> width = 2.4 units!
    # Arrow 3: System -> TPS (Rightward arrow)
    arrow3 = FancyArrowPatch((8.75, 2.15), (10.85, 2.15),
                             arrowstyle='-|>', mutation_scale=20,
                             color="#15803D", linewidth=2.6, zorder=4)
    ax.add_patch(arrow3)
    ax.text(9.8, 2.48, "Synthetic CT (sCT)\nCalibrated Volume [HU]", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color="#15803D", zorder=5)

    # Diagram Title
    ax.text(7.1, 4.15, "DFD Level 0: Context-Level System Architecture",
            ha="center", va="center", fontsize=13.5, fontweight="bold", color="#1E293B")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

def create_dfd_level_1(output_path="report/images/dfd_level_1_pipeline.png"):
    # Generous width of 19.0 inches to allow 1.6-unit gaps between all 5 boxes
    fig, ax = plt.subplots(figsize=(18.5, 4.8), dpi=300)
    ax.set_xlim(0, 18.5)
    ax.set_ylim(0, 4.8)
    ax.axis('off')

    # Card background
    bg = FancyBboxPatch((0.15, 0.15), 18.2, 4.5, boxstyle="round,pad=0.1,rounding_size=0.18",
                         facecolor="#FAFAFA", edgecolor="#D6D6D6", linewidth=1.5, zorder=0)
    ax.add_patch(bg)

    # Layout:
    # Box 1: [0.6, 2.8] (w=2.2)
    # Gap 1: [2.8, 4.4] (gap=1.6)
    # Box 2: [4.4, 6.6] (w=2.2)
    # Gap 2: [6.6, 8.2] (gap=1.6)
    # Box 3: [8.2, 10.9] (w=2.7) [Central Engine]
    # Gap 3: [10.9, 12.5] (gap=1.6)
    # Box 4: [12.5, 14.7] (w=2.2)
    # Gap 4: [14.7, 16.3] (gap=1.6)
    # Box 5: [16.3, 18.5] -> shift everything slightly left by 0.3
    
    stages = [
        (0.5, 1.1, 2.2, 2.2, "1.0 Ingestion\n& Masking", "NIfTI / DICOM Read\nBrain Mask Extr.", "#EFF6FF", "#1D4ED8", "#1E3A8A"),
        (4.1, 1.1, 2.2, 2.2, "2.0 Intensity\nScaling", "Clip [-1000, 1500]\nRescale to [-1, 1]", "#EEF2FF", "#4338CA", "#312E81"),
        (7.7, 0.95, 2.7, 2.5, "3.0 Generative\nEngine", "Stage-1 ResUNet +\nFiLM Residual Diff.", "#FFFBEB", "#D97706", "#92400E"),
        (11.8, 1.1, 2.2, 2.2, "4.0 Physical\nRe-Calibration", "De-normalize [-1, 1]\nto Physical HU", "#F0FDFA", "#0F766E", "#134E4A"),
        (15.4, 1.1, 2.2, 2.2, "5.0 Clinical\nEvaluation", "SSIM, PSNR, MAE\nBone Dice (>300 HU)", "#F0FDF4", "#15803D", "#14532D")
    ]

    for x, y, w, h, title, sub, bg_c, bdr_c, txt_c in stages:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.14",
                             facecolor=bg_c, edgecolor=bdr_c, linewidth=2.2 if "3.0" not in title else 2.8, zorder=2)
        ax.add_patch(box)
        # Title
        ax.text(x + w/2, y + h*0.68, title, ha="center", va="center",
                fontsize=11.5 if "3.0" not in title else 12.5, fontweight="bold", color=txt_c, zorder=3)
        # Divider line
        ax.plot([x + 0.15, x + w - 0.15], [y + h*0.42, y + h*0.42], color=bdr_c, linewidth=0.9, alpha=0.4, zorder=3)
        # Subtitle
        ax.text(x + w/2, y + h*0.22, sub, ha="center", va="center",
                fontsize=8.5 if "3.0" not in title else 9.2, color="#334155", zorder=3)

    # Connections between boxes
    # Arrow 1: Box 1 (end=2.7) -> Box 2 (start=4.1), gap = 1.4 units!
    a1 = FancyArrowPatch((2.75, 2.2), (4.05, 2.2), arrowstyle='-|>', mutation_scale=18,
                         color="#1D4ED8", linewidth=2.4, zorder=4)
    ax.add_patch(a1)
    ax.text(3.4, 2.5, "Raw MRI\n& Mask", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#1D4ED8")

    # Arrow 2: Box 2 (end=6.3) -> Box 3 (start=7.7), gap = 1.4 units!
    a2 = FancyArrowPatch((6.35, 2.2), (7.65, 2.2), arrowstyle='-|>', mutation_scale=18,
                         color="#4338CA", linewidth=2.4, zorder=4)
    ax.add_patch(a2)
    ax.text(7.0, 2.5, r"$\mathbf{x} \in [-1, 1]$" + "\nNormalized", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#4338CA")

    # Arrow 3: Box 3 (end=10.4) -> Box 4 (start=11.8), gap = 1.4 units!
    a3 = FancyArrowPatch((10.45, 2.2), (11.75, 2.2), arrowstyle='-|>', mutation_scale=18,
                         color="#D97706", linewidth=2.4, zorder=4)
    ax.add_patch(a3)
    ax.text(11.1, 2.5, r"$\mathbf{\hat{y}}_{\mathrm{norm}}$" + "\nPredicted", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#B45309")

    # Arrow 4: Box 4 (end=14.0) -> Box 5 (start=15.4), gap = 1.4 units!
    a4 = FancyArrowPatch((14.05, 2.2), (15.35, 2.2), arrowstyle='-|>', mutation_scale=18,
                         color="#0F766E", linewidth=2.4, zorder=4)
    ax.add_patch(a4)
    ax.text(14.7, 2.5, r"$\mathbf{\hat{y}}_{\mathrm{HU}}$" + "\nCalibrated", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#0F766E")

    # Diagram Title
    ax.text(9.25, 4.25, "DFD Level 1: Subsystem Processing Pipeline Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#1E293B")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    create_dfd_level_0()
    create_dfd_level_1()
