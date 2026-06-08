"""% change (ANE vs PLA) across bouts — all significant metrics on one axis.

Normalises each subject's bout value as:
    pct = (ANE - PLA) / PLA * 100

Puts amplitude, power, and regularity metrics on the same scale so their
magnitudes and washout rates can be compared directly.
Saved to overleaf/figures/suppl/.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import OUT
from fig_style import RC, FIGSIZE, SUPPL, save_fig

BOUTS = [1, 2, 3, 4]
BOUT_LABELS = ["Bout 1\n(~3 min)", "Bout 2\n(~8 min)",
               "Bout 3\n(~12 min)", "Bout 4\n(~17 min)"]

# Significant metrics (FDR < 0.05 at bout 1).
# Colors: qualitative palette that does NOT overlap with the condition palette
# (ANEC=#009E73 green, PLAC=#CC79A7 lilac) to avoid cross-figure confusion.
METRICS = [
    ("med_amp",  "Median amp.",      "#1f77b4"),   # steel blue
    ("rms_amp",  "RMS amp.",         "#ff7f0e"),   # orange
    ("p_20_60",  "Power 20–60 Hz",   "#d62728"),   # red
    ("p_60_150", "Power 60–150 Hz",  "#9467bd"),   # purple
    ("cv_ipi",   "CV of IPI",        "#8c564b"),   # brown
]


def pct_change(df):
    """Return DataFrame with % change per subject × feat × bout."""
    rows = []
    for subj, g in df.groupby("subj"):
        ane = g[g.cond == "ANE"].set_index("bout")
        pla = g[g.cond == "PLA"].set_index("bout")
        common = ane.index.intersection(pla.index)
        for bout in common:
            for feat, _, _ in METRICS:
                a, p = ane.loc[bout, feat], pla.loc[bout, feat]
                if p != 0 and np.isfinite(a) and np.isfinite(p):
                    rows.append(dict(subj=subj, bout=bout, feat=feat,
                                     pct=(a - p) / abs(p) * 100))
    return pd.DataFrame(rows)


def main():
    df  = pd.read_csv(OUT / "bout_features.csv")
    pct = pct_change(df)

    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    xs = np.arange(len(BOUTS))
    for feat, label, color in METRICS:
        sub = pct[pct.feat == feat]
        means, sems = [], []
        for b in BOUTS:
            v = sub[sub.bout == b]["pct"].dropna().values
            means.append(v.mean())
            sems.append(v.std() / np.sqrt(len(v)))
        means, sems = np.array(means), np.array(sems)
        ax.plot(xs, means, color=color, lw=1.8, marker="o", ms=5, label=label)
        ax.fill_between(xs, means - sems, means + sems, color=color, alpha=0.15)

    ax.axhline(0, color="#555555", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(BOUT_LABELS, fontsize=7.5)
    ax.set_xlabel("Chewing bout")
    ax.set_ylabel("Change vs. placebo (%)")
    ax.legend(fontsize=7.5, loc="upper right")

    n = pct["subj"].nunique()
    ax.text(0.98, 0.05, f"N = {n}", transform=ax.transAxes,
            fontsize=7.5, ha="right", va="bottom", color="0.4")

    fig.tight_layout()
    save_fig(fig, "s_pct_change", SUPPL)


if __name__ == "__main__":
    main()
