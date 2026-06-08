"""Experimental paradigm figure — timeline diagram.

Within-session crossover: ANE and PLA blocks separated by ~4 min (validated
as sufficient for washout). Both blocks in the same session.
Saved to overleaf/figures/suppl/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fig_style import RC, FIGSIZE, ANEC, PLAC, SUPPL, save_fig

BOUT_ONSETS = [3.0, 7.5, 12.0, 17.0]
BOUT_DUR    = 1.0   # 60 s
ROW_H       = 0.50
Y_ANE       = 1.0
Y_PLA       = 0.0


def draw_session(ax, y, color, label):
    t_end = BOUT_ONSETS[-1] + BOUT_DUR + 1.0

    # Background track
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, y - ROW_H / 2), t_end, ROW_H,
        boxstyle="round,pad=0.04", linewidth=0,
        facecolor="#F5F5F5", zorder=1))

    # Condition label
    ax.text(-0.5, y, label, ha="right", va="center",
            fontsize=9, fontweight="bold", color=color)

    # Bout blocks
    for i, onset in enumerate(BOUT_ONSETS):
        ax.add_patch(mpatches.Rectangle(
            (onset, y - ROW_H / 2 + 0.04), BOUT_DUR, ROW_H - 0.08,
            linewidth=0.8, edgecolor="white", facecolor=color,
            zorder=4, alpha=0.90))
        ax.text(onset + BOUT_DUR / 2, y, f"B{i+1}",
                ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold", zorder=5)

    # Cognitive task labels between bouts
    for i in range(len(BOUT_ONSETS) - 1):
        mid = BOUT_ONSETS[i] + BOUT_DUR + \
              (BOUT_ONSETS[i+1] - BOUT_ONSETS[i] - BOUT_DUR) / 2
        ax.text(mid, y, "cognitive\ntask", ha="center", va="center",
                fontsize=6.0, color="#888888", zorder=3)


def main():
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    draw_session(ax, Y_ANE, ANEC, "Anesthesia")
    draw_session(ax, Y_PLA, PLAC, "Placebo")

    # ── Single shared spray line (black, passes through both rows) ─────────────
    ax.axvline(0, color="black", lw=1.2, ls="--", zorder=6,
               ymin=0.12, ymax=0.95)
    # Label vertical, centrado en la zona de los Arm labels, con fondo blanco
    y_arm_center = Y_ANE + ROW_H / 2 + 0.28
    ax.text(-1.5, y_arm_center, "Spray (t = 0)",
            ha="center", va="center", fontsize=7.5,
            color="black", fontweight="bold", rotation=90, zorder=10)

    # ── Inter-block interval annotation ────────────────────────────────────────
    y_mid = (Y_ANE - ROW_H / 2 + Y_PLA + ROW_H / 2) / 2
    ax.annotate("", xy=(0.5, Y_PLA + ROW_H / 2),
                xytext=(0.5, Y_ANE - ROW_H / 2),
                arrowprops=dict(arrowstyle="<->", lw=0.9,
                                color="#666666"))
    ax.text(1.5, y_mid, "~4 min\ninter-block", ha="left", va="center",
            fontsize=6.5, color="#666666")

    # ── Arm labels at the top ──────────────────────────────────────────────────
    y_top = Y_ANE + ROW_H / 2 + 0.38
    # Arm 1: ANE → PLA
    ax.text(4.0, y_top, "Arm 1 (n=16): ", ha="right", va="center",
            fontsize=7.5, color="#333333")
    ax.text(4.0, y_top, "Anesthesia", ha="left", va="center",
            fontsize=7.5, color=ANEC, fontweight="bold")
    ax.text(7.8, y_top, "→  Placebo", ha="left", va="center",
            fontsize=7.5, color=PLAC, fontweight="bold")

    # Arm 2: PLA → ANE
    ax.text(4.0, y_top - 0.20, "Arm 2 (n=18): ", ha="right", va="center",
            fontsize=7.5, color="#333333")
    ax.text(4.0, y_top - 0.20, "Placebo", ha="left", va="center",
            fontsize=7.5, color=PLAC, fontweight="bold")
    ax.text(7.8, y_top - 0.20, "→  Anesthesia", ha="left", va="center",
            fontsize=7.5, color=ANEC, fontweight="bold")

    # ── Axes ───────────────────────────────────────────────────────────────────
    ax.set_xlim(-3.0, 20.0)
    ax.set_ylim(-0.50, Y_ANE + ROW_H / 2 + 0.75)
    ax.set_xlabel("Time post-application (min)")
    ax.set_xticks([0] + BOUT_ONSETS)
    ax.set_xticklabels(["0", "3", "7.5", "12", "17"], fontsize=8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "s0_paradigm", SUPPL)


if __name__ == "__main__":
    main()
