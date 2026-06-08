"""Supplementary bout time-course line plots (S9 — one panel per metric).

ANE vs PLA mean ± SEM across bouts 1–4.
Outputs → overleaf/figures/suppl/s9_bout_{feat}.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import OUT
from fig_style import RC, FIGSIZE, ANEC, PLAC, SUPPL, save_fig

BOUT_LABELS = ["B1\n(~3 min)", "B2\n(~8 min)", "B3\n(~12 min)", "B4\n(~17 min)"]

METRICS = [
    ("med_amp",   r"Median cycle amplitude ($\mu$V)",    1e6),
    ("rms_amp",   r"RMS amplitude ($\mu$V)",             1e6),
    ("p_60_150",  r"Power 60–150 Hz ($\mu$V$^2$/Hz)",   1e12),
    ("cv_ipi",    "CV of inter-peak interval",           1.0),
    ("MDF",       "Median frequency (Hz)",               1.0),
    ("chew_rate", "Chewing rate (Hz)",                   1.0),
]


def plot_bout_metric(feat, ylabel, scale, df, st):
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for cond, color, lab in [("ANE", ANEC, "Anesthesia"),
                              ("PLA", PLAC, "Placebo")]:
        g     = df[df.cond == cond].groupby("bout")[feat]
        mu    = g.mean() * scale
        se    = g.sem()  * scale
        bouts = mu.index.values
        ax.plot(bouts, mu.values, "o-", color=color, lw=1.6, ms=5.5, label=lab)
        ax.fill_between(bouts, (mu - se).values, (mu + se).values,
                        color=color, alpha=0.20)
        ax.errorbar(bouts, mu.values, yerr=se.values,
                    fmt="none", color=color, capsize=3, elinewidth=0.8)

    if feat in st.index:
        pfdr = st.loc[feat, "p_b1_fdr"]
        if pfdr < 0.05:
            sym = "**" if pfdr < 0.01 else "*"
            ylo, yhi = ax.get_ylim()
            ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.14)
            yhi2 = ax.get_ylim()[1]
            ax.text(1, yhi2 - (yhi2 - ylo) * 0.03, sym,
                    ha="center", va="top", fontsize=12, color="black")

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(BOUT_LABELS, fontsize=8)
    ax.set_xlabel("Chewing bout")
    ax.set_ylabel(ylabel)
    ax.legend(handles=[mpatches.Patch(color=ANEC, label="Anesthesia"),
                       mpatches.Patch(color=PLAC, label="Placebo")],
              loc="best")
    fig.tight_layout()
    save_fig(fig, f"s9_bout_{feat}", SUPPL)


def main():
    df = pd.read_csv(OUT / "bout_features.csv")
    st = pd.read_csv(OUT / "stats_bout.csv").set_index("feature")
    for feat, ylabel, scale in METRICS:
        plot_bout_metric(feat, ylabel, scale, df, st)
    print("Done.")


if __name__ == "__main__":
    main()
