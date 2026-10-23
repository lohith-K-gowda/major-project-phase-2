import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Polygon

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

def create_dfd_level_2(output_path="report/images/dfd_level_2_ddim_loop.png"):
    fig, ax = plt.subplots(figsize=(8.6, 9.0), dpi=300)
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, 9.0)
    ax.axis('off')

    # Card background
    bg = FancyBboxPatch((0.15, 0.15), 8.3, 8.7, boxstyle="round,pad=0.15,rounding_size=0.18",
                         facecolor="#FAFAFA", edgecolor="#D6D6D6", linewidth=1.5, zorder=0)
    ax.add_patch(bg)

    # Title
    ax.text(4.3, 8.5, "DFD Level 2: DDIM 50-Step Deterministic Reverse Sampling Loop",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#1E293B")

    # 1. Start Node: Gaussian Noise
    # Ellipse center at (4.3, 7.6), width=5.6, height=1.0
    start_ellipse = Ellipse((4.3, 7.6), width=5.6, height=1.0,
                            facecolor="#F3F4F6", edgecolor="#4B5563", linewidth=2.2, zorder=2)
    ax.add_patch(start_ellipse)
    ax.text(4.3, 7.6, r"Gaussian Noise  $\mathbf{y}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#111827", zorder=3)

    # Arrow 1: Noise -> Time Step
    a1 = FancyArrowPatch((4.3, 7.1), (4.3, 6.55), arrowstyle='-|>', mutation_scale=18,
                         color="#374151", linewidth=2.2, zorder=4)
    ax.add_patch(a1)

    # 2. Node 2: Time Step t = \tau_i
    # Center at (4.3, 6.05), w=3.6, h=0.9
    box_time = FancyBboxPatch((2.5, 5.6), 3.6, 0.9, boxstyle="round,pad=0.08,rounding_size=0.12",
                              facecolor="#FEF3C7", edgecolor="#D97706", linewidth=2.2, zorder=2)
    ax.add_patch(box_time)
    ax.text(4.3, 6.05, r"Time Step  $t = \tau_i$", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="#92400E", zorder=3)

    # Arrow 2: Time Step -> Denoiser
    a2 = FancyArrowPatch((4.3, 5.6), (4.3, 5.05), arrowstyle='-|>', mutation_scale=18,
                         color="#374151", linewidth=2.2, zorder=4)
    ax.add_patch(a2)

    # 3. Node 3: Denoiser \epsilon_\theta(y_t, t, x, m)
    # Center at (4.3, 4.55), w=4.8, h=0.95
    box_denoiser = FancyBboxPatch((1.9, 4.07), 4.8, 0.95, boxstyle="round,pad=0.1,rounding_size=0.14",
                                  facecolor="#EEF2FF", edgecolor="#4F46E5", linewidth=2.4, zorder=2)
    ax.add_patch(box_denoiser)
    ax.text(4.3, 4.55, r"Denoiser  $\epsilon_\theta(\mathbf{y}_t, \, t, \, \mathbf{x}, \, \mathbf{m})$",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#312E81", zorder=3)

    # Arrow 3: Denoiser -> DDIM Update
    a3 = FancyArrowPatch((4.3, 4.07), (4.3, 3.52), arrowstyle='-|>', mutation_scale=18,
                         color="#374151", linewidth=2.2, zorder=4)
    ax.add_patch(a3)

    # 4. Node 4: DDIM Update y_{\tau_{i-1}}
    # Center at (4.3, 3.05), w=4.0, h=0.9
    box_update = FancyBboxPatch((2.3, 2.6), 4.0, 0.9, boxstyle="round,pad=0.08,rounding_size=0.12",
                                facecolor="#FFFBEB", edgecolor="#CA8A04", linewidth=2.2, zorder=2)
    ax.add_patch(box_update)
    ax.text(4.3, 3.05, r"DDIM Update  $\mathbf{y}_{\tau_{i-1}}$", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="#854D0E", zorder=3)

    # Arrow 4: DDIM Update -> Condition
    a4 = FancyArrowPatch((4.3, 2.6), (4.3, 1.95), arrowstyle='-|>', mutation_scale=18,
                         color="#374151", linewidth=2.2, zorder=4)
    ax.add_patch(a4)

    # 5. Node 5: Condition Diamond (i = 1 ?)
    # Center at (4.3, 1.35)
    diamond_pts = [[3.1, 1.35], [4.3, 1.95], [5.5, 1.35], [4.3, 0.75]]
    diamond = Polygon(diamond_pts, closed=True, facecolor="#FCE7F3", edgecolor="#DB2777",
                      linewidth=2.4, zorder=2)
    ax.add_patch(diamond)
    ax.text(4.3, 1.35, r"$i = 1$ ?", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#831843", zorder=3)

    # 6. Branch "Yes": Diamond -> Synthesized CT y_0
    a_yes = FancyArrowPatch((5.5, 1.35), (6.35, 1.35), arrowstyle='-|>', mutation_scale=18,
                            color="#059669", linewidth=2.4, zorder=4)
    ax.add_patch(a_yes)
    ax.text(5.92, 1.6, "Yes", ha="center", va="bottom",
            fontsize=11.5, fontweight="bold", color="#059669", zorder=5)

    # Node 6: Synthesized CT y_0
    box_out = FancyBboxPatch((6.4, 0.85), 1.9, 1.0, boxstyle="round,pad=0.1,rounding_size=0.14",
                             facecolor="#ECFDF5", edgecolor="#059669", linewidth=2.2, zorder=2)
    ax.add_patch(box_out)
    ax.text(7.35, 1.35, "Synthesized CT\n" + r"$\mathbf{y}_0$", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#065F46", zorder=3)

    # 7. Branch "No (i <- i - 1)": Loop back to Time Step
    # Line from left vertex (3.1, 1.35) -> left to x=1.1
    # Then up to y=6.05
    # Then arrow right to x=2.5 (Time Step left border)
    ax.plot([3.1, 1.1], [1.35, 1.35], color="#BE185D", linewidth=2.2, zorder=4)
    ax.plot([1.1, 1.1], [1.35, 6.05], color="#BE185D", linewidth=2.2, zorder=4)
    a_loop = FancyArrowPatch((1.1, 6.05), (2.45, 6.05), arrowstyle='-|>', mutation_scale=18,
                             color="#BE185D", linewidth=2.2, zorder=4)
    ax.add_patch(a_loop)

    # Text label for loop
    ax.text(1.0, 3.7, r"No  $(i \leftarrow i - 1)$", ha="right", va="center",
            fontsize=11, fontweight="bold", color="#9D174D", zorder=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    create_dfd_level_2()
