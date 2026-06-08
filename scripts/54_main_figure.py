"""Main result figure — three SEPARATE panels for Inkscape assembly:
  fig2a_med_amp.png    Median cycle amplitude by bout  (ANE vs PLA, boxplot)
  fig2b_cv_ipi.png     CV of inter-peak interval by bout
  fig2c_heatmap.png    Effect-size heatmap  (dz × metrics × bouts)

All panels share the same figsize, DPI, and RC (from fig_style).
Saved to overleaf/figures/main/.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from config import OUT
from fig_style import RC, FIGSIZE, ANEC, PLAC, ANED, PLAD, MAIN, save_fig

LAB  = {"ANE": "Anesthesia", "PLA": "Placebo"}

AMP_FEATS  = ["med_amp", "rms_amp", "totpow", "p_20_60", "p_60_150", "cv_ipi"]
TIME_FEATS = ["MDF", "MNF", "chew_rate", "duty"]
HMAP_ORDER = AMP_FEATS + TIME_FEATS

FEAT_LABELS = {
    "med_amp":   "Median amp.",
    "rms_amp":   "RMS amp.",
    "totpow":    "Total power",
    "p_20_60":   "Power 20–60 Hz",
    "p_60_150":  "Power 60–150 Hz",
    "cv_ipi":    "CV of IPI",
    "MDF":       "MDF",
    "MNF":       "MNF",
    "chew_rate": "Chewing rate",
    "duty":      "Duty cycle",
}

SCALE  = {"med_amp": 1e6, "rms_amp": 1e6}
YLABEL = {
    "med_amp": "Median cycle amplitude (µV)",
    "cv_ipi":  "CV of inter-peak interval",
}
BOUT_LABELS = ["B1\n(~3 min)", "B2\n(~8 min)", "B3\n(~12 min)", "B4\n(~17 min)"]


def _save(fig, stem):
    save_fig(fig, stem, MAIN)


# ── Panel A / B — bout boxplot ────────────────────────────────────────────────

def panel_boxplot(feat, stem, df, stats):
    plt.rcParams.update(RC)
    sc = SCALE.get(feat, 1.0)
    d  = df.copy()
    d[feat]        = d[feat] * sc
    d["Condition"] = d["cond"].map(LAB)
    hue_order = [LAB["ANE"], LAB["PLA"]]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.boxplot(
        data=d, x="bout", y=feat, hue="Condition",
        palette=[ANEC, PLAC], hue_order=hue_order,
        showfliers=False, width=0.52, linewidth=0.8, ax=ax,
    )
    sns.stripplot(
        data=d, x="bout", y=feat, hue="Condition",
        palette=[ANED, PLAD], hue_order=hue_order,
        dodge=True, size=2.2, alpha=0.45, ax=ax, legend=False,
    )

    if feat in stats.index:
        pfdr = stats.loc[feat, "p_b1_fdr"]
        if pfdr < 0.05:
            sig = "**" if pfdr < 0.01 else "*"
            ylo, yhi = ax.get_ylim()
            ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.14)
            ax.annotate(sig,
                        xy=(0, 0.97), xycoords=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=11, color="black")

    ax.set_xlabel("Chewing bout")
    ax.set_ylabel(YLABEL.get(feat, feat))
    ax.set_xticks(range(4))
    ax.set_xticklabels(BOUT_LABELS, fontsize=7.5)
    if ax.get_legend():
        ax.get_legend().remove()

    handles = [mpatches.Patch(color=ANEC, label="Anesthesia"),
               mpatches.Patch(color=PLAC, label="Placebo")]
    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.0)

    fig.tight_layout()
    _save(fig, stem)


# ── Panel C — effect-size heatmap ─────────────────────────────────────────────

def panel_heatmap(stats):
    plt.rcParams.update(RC)
    feat_order = [f for f in HMAP_ORDER if f in stats.index]
    labels     = [FEAT_LABELS.get(f, f) for f in feat_order]
    n_total    = len(feat_order)
    n_amp      = sum(1 for f in feat_order if f in AMP_FEATS)
    n_time     = n_total - n_amp

    dz   = stats.loc[feat_order, ["dz_b1", "dz_b2", "dz_b3", "dz_b4"]].values.astype(float)
    pfdr = stats.loc[feat_order, "p_b1_fdr"].values

    fig, ax = plt.subplots(figsize=FIGSIZE)

    im = ax.imshow(dz, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")

    # FDR asterisks in bout-1 column only
    for row, p in enumerate(pfdr):
        if p < 0.05:
            sym = "**" if p < 0.01 else "*"
            ax.text(0, row, sym, ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")

    # Dashed separator between amplitude/regularity and timing blocks
    if 0 < n_amp < n_total:
        ax.axhline(n_amp - 0.5, color="white", lw=1.5, ls="--")

    # y-tick labels: amplitude = black, timing = gray
    ax.set_yticks(range(n_total))
    ax.set_yticklabels(labels, fontsize=8)
    for i, tick in enumerate(ax.yaxis.get_ticklabels()):
        tick.set_color("black" if i < n_amp else "#666666")

    ax.set_xticks(range(4))
    ax.set_xticklabels(["Bout 1", "Bout 2", "Bout 3", "Bout 4"], fontsize=8.5)
    ax.set_xlabel("Bout (washout time course)")

    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.90)
    cb.ax.set_ylabel("Cohen's $d_z$", fontsize=8, rotation=270, labelpad=14)
    cb.ax.tick_params(labelsize=7)

    fig.tight_layout(pad=1.0, rect=[0.07, 0, 1, 1])

    # Group labels in figure fraction — placed after tight_layout so axes position is known
    # For imshow origin='upper': row i centre → axes y_frac = 1 - (i + 0.5) / n_total
    pos = ax.get_position()
    y_amp_frac  = 1.0 - ((n_amp  - 1) / 2.0 + 0.5) / n_total
    y_time_frac = 1.0 - (n_amp + (n_time - 1) / 2.0 + 0.5) / n_total
    y_amp_fig   = pos.y0 + y_amp_frac  * pos.height
    y_time_fig  = pos.y0 + y_time_frac * pos.height
    x_gl = 0.02   # fixed far-left position, clear of all tick labels
    kw = dict(ha="center", va="center", fontsize=7, rotation=90,
              transform=fig.transFigure)
    fig.text(x_gl, y_amp_fig,  "Amplitude & regularity", color="black",   **kw)
    fig.text(x_gl, y_time_fig, "Timing & frequency",     color="#666666", **kw)

    _save(fig, "fig2c_heatmap")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(OUT / "bout_features.csv")
    st = pd.read_csv(OUT / "stats_bout.csv").set_index("feature")

    panel_boxplot("med_amp", "fig2a_med_amp", df, st)
    panel_boxplot("cv_ipi",  "fig2b_cv_ipi",  df, st)
    panel_heatmap(st)
    print("All panels saved to outputs/ and overleaf/figures/")


if __name__ == "__main__":
    main()
