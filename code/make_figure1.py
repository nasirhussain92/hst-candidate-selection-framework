import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6.5)
ax.axis('off')

def box(x, y, w, h, text, fc="#f2f2f2", ec="#333333", fontsize=10, weight="normal", ls="-"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                        linewidth=1.4, edgecolor=ec, facecolor=fc, linestyle=ls)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, wrap=True)
    return (x, y, w, h)

def arrow(x1, y1, x2, y2, style="-|>", color="#333333", lw=1.4, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                         mutation_scale=14, linewidth=lw, color=color, linestyle=ls)
    ax.add_patch(a)

# --- Input layer ---
b_static = box(0.3, 4.7, 2.1, 1.1, "Static Features\n$\\mathbf{x}_{static}$\n(resume, CQ, AQ)\nEq. 7", fc="#eaf2fb")
b_L = box(0.3, 2.9, 2.1, 1.1, "Assessment-Performance\nTrajectory  $\\mathbf{L}$\n(multi-stage scores)\n\u00a73.3", fc="#eaf7ef")
b_S = box(0.3, 1.1, 2.1, 1.1, "Sentiment Trajectory\n$\\mathbf{S}$\n(affect across interview\nstages)   Eq. 1 / 6", fc="#fdf1e6")

# --- Shape-feature extractor ---
b_phi = box(3.4, 2.0, 2.3, 2.0,
            "Shape-Feature\nExtractor  $\\Phi(\\cdot)$\n[slope, volatility,\nrecovery, consistency]\nEq. 2\u20133\n\n(sequence-agnostic;\napplied identically to\n$\\mathbf{L}$ and $\\mathbf{S}$)",
            fc="#ffffff", fontsize=9.5)

# --- Fusion ---
b_fuse = box(6.7, 2.0, 2.2, 2.0,
             "Linear Fusion\n$\\sigma(\\mathbf{w}^\\top[\\;\\mathbf{x}_{static},\\;\\Phi(\\mathbf{L}),\\;\\Phi(\\mathbf{S})\\;])$\nEq. 4\n\n(implemented &\nempirically tested)",
             fc="#fff6e8", fontsize=9.5)

# --- Output ---
b_hcs = box(9.6, 2.6, 1.1, 0.9, "HCS\nscore", fc="#eaf7ef", weight="bold")

# --- Proposed (not implemented) neural path, dashed ---
b_neural = box(6.7, 4.6, 2.2, 1.1,
               "Proposed Cross-Attention\nFusion (Eq. 5)\n\u2014 not implemented \u2014\napproximated by MLP\nbenchmark, \u00a75.3",
               fc="#f7f7f7", ec="#999999", fontsize=8.7, ls="--")

# arrows: inputs -> phi (L and S only)
arrow(2.4, 3.45, 3.4, 3.0)   # L -> Phi
arrow(2.4, 1.65, 3.4, 2.4)   # S -> Phi
# static bypasses Phi, goes straight to fusion
arrow(2.4, 5.25, 6.7, 3.6, color="#4472c4")
# phi -> fusion
arrow(5.7, 3.0, 6.7, 3.0)
# fusion -> HCS
arrow(8.9, 3.0, 9.6, 3.05)
# static + phi -> neural (dashed, alternative path)
arrow(2.4, 5.15, 6.7, 5.1, ls="--", color="#999999")
arrow(5.7, 3.6, 6.7, 4.9, ls="--", color="#999999")

ax.text(5.5, 6.15, "Figure 1. Hybrid Sentiment\u2013Trajectory (HST) Framework: Model Architecture",
        ha="center", fontsize=12, weight="bold")
ax.text(5.5, 0.35,
        "Solid path = implemented linear fusion (Eq. 4), evaluated in \u00a75\u2013\u00a77.   "
        "Dashed path = proposed neural fusion (Eq. 5), not implemented; approximated by an MLP benchmark (\u00a75.3).",
        ha="center", fontsize=8.5, style="italic", color="#555555")

plt.tight_layout()
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(_THIS_DIR, "..", "figures", "figure1_framework.png"), dpi=220, bbox_inches="tight")
print("saved")
