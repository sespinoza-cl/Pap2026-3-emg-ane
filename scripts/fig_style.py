"""Shared figure style — Paper3 (EMG masticatorio bajo anestesia).

All figure scripts import RC, FIGSIZE*, ANEC, PLAC and save_fig from here.
Rules enforced:
  - No ax.set_title() in any figure script
  - Axis labels fontsize=10, fontweight='bold' (via RC axes.labelsize/weight)
  - All panels: same DPI (200), same RC
  - Main figures  → MAIN   = outputs/main/
  - Suppl figures → SUPPL  = outputs/suppl/
  - Canonical paper figures (fig1.png, sup1-3.png) live in overleaf/figures/
    and are assembled manually in Inkscape — NOT written by these scripts.
"""
import matplotlib.pyplot as plt
from config import PROJ

MAIN  = PROJ / "outputs" / "main"
SUPPL = PROJ / "outputs" / "suppl"

for _d in (MAIN, SUPPL):
    _d.mkdir(parents=True, exist_ok=True)

# ── standard panel size — ALL panels use this so Inkscape grids align ────────
FIGSIZE      = (5.2, 4.6)    # single size for every panel
FIGSIZE_WIDE = FIGSIZE        # kept for import compatibility

# ── paper palette (Okabe-Ito — colorblind-safe) ───────────────────────────────
ANEC = "#009E73"   # bluish-green — Anesthesia
PLAC = "#CC79A7"   # reddish-purple / lilac — Placebo
ANED = "#007A5A"   # darker green  (strip / scatter dots)
PLAD = "#AA5F88"   # darker lilac

# ── rcParams applied by every script ─────────────────────────────────────────
RC = {
    "figure.dpi":         200,
    "savefig.dpi":        200,
    "font.size":          9,
    "font.family":        "sans-serif",
    "axes.linewidth":     0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     10,      # large axis labels
    "axes.labelweight":   "bold",
    "xtick.major.width":  0.9,
    "ytick.major.width":  0.9,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "legend.frameon":     False,
    "legend.fontsize":    8,
}


def save_fig(fig, stem, folder):
    """Save panel to outputs/main/ or outputs/suppl/ (no flat-outputs duplicate)."""
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", stem)
