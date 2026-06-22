"""Washout analysis — two separate panels for Inkscape assembly:
  fig3a_washout_curve.png   ANE-PLA decay over 4 bouts (mean ± 95% CI)
  fig3b_individual.png      Per-subject bar plot at bout 1
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from config import OUT
from fig_style import RC, FIGSIZE, ANEC, PLAC, MAIN, save_fig

MIN_POST = np.array([3.3, 7.9, 12.6, 17.2])


def _load():
    df = pd.read_csv(OUT / "bout_features.csv")
    feat = "med_amp"
    a = df[df.cond == "ANE"].pivot_table(index="subj", columns="bout", values=feat)
    p = df[df.cond == "PLA"].pivot_table(index="subj", columns="bout", values=feat)
    common = a.dropna().index.intersection(p.dropna().index)
    a, p = a.loc[common], p.loc[common]
    d = (a - p) * 1e6
    n = len(common)
    mu, se = d.mean().values, d.sem().values

    rng = np.random.default_rng(0)
    bt  = np.array([d.values[rng.integers(0, n, n)].mean(0) for _ in range(2000)])
    ci_lo = np.percentile(bt, 2.5,  axis=0)
    ci_hi = np.percentile(bt, 97.5, axis=0)

    slopes  = np.array([np.polyfit(MIN_POST, d.loc[s].values, 1)[0] for s in common])
    trend_p = stats.wilcoxon(slopes).pvalue
    b1      = d[1].values
    b1_p    = stats.wilcoxon(b1).pvalue
    frac    = np.mean(b1 > 0) * 100

    print(f"N={n}, bout1: Wilcoxon p={b1_p:.4f}, {frac:.0f}% ANE>PLA, trend p={trend_p:.4f}")

    return dict(mu=mu, se=se, ci_lo=ci_lo, ci_hi=ci_hi,
                b1=b1, b1_p=b1_p, frac=frac, n=n)


def panel_curve(wd):
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.fill_between(MIN_POST, wd["ci_lo"], wd["ci_hi"], color=ANEC, alpha=0.22)
    ax.errorbar(MIN_POST, wd["mu"], fmt="o", color=ANEC, ms=6, zorder=4,
                capsize=3, elinewidth=0.9)

    ax.axhline(0, color="#555555", ls="--", lw=0.9)
    ax.set_xlabel("Time post-application (min)")
    ax.set_ylabel("ANE $-$ PLA amplitude (µV)")
    fig.tight_layout()
    save_fig(fig, "fig3a_washout_curve", MAIN)


def panel_individual(wd):
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    RESP_C = ANEC       # responders (ANE > PLA) — same treatment color
    NONR_C = "#AAAAAA"  # non-responders — neutral gray (≠ condition palette)

    order  = np.argsort(wd["b1"])
    colors = [RESP_C if v > 0 else NONR_C for v in wd["b1"][order]]
    ax.bar(range(wd["n"]), wd["b1"][order], color=colors, width=0.85, linewidth=0)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks([])
    ax.set_xlabel("Participants (sorted by effect)")
    ax.set_ylabel("ANE $-$ PLA amplitude, bout 1 (µV)")
    ax.legend(handles=[mpatches.Patch(color=RESP_C, label="ANE > PLA"),
                       mpatches.Patch(color=NONR_C, label="ANE < PLA")],
              loc="lower right")
    fig.tight_layout()
    save_fig(fig, "fig3b_individual", MAIN)


def main():
    wd = _load()
    panel_curve(wd)
    panel_individual(wd)


if __name__ == "__main__":
    main()
