"""S5 revised: spectral fatigue slope (MDF/MNF slope, Hz/min) per chewing bout.

Replaces the window-by-window timecourse (uninterpretable due to masticatory gating)
with the linear regression slope of MDF/MNF within each 60-s bout,
stratified by bout and condition. A null, identical slope across conditions
demonstrates the absence of differential spectral fatigue at any point in the experiment.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import OUT, DERIV
from emg_tf import reshape_bouts, mdf_mnf_timecourse

from fig_style import RC, FIGSIZE, ANEC as ANE_C, PLAC as PLA_C, SUPPL, save_fig
plt.rcParams.update(RC)

BOUTS = [1, 2, 3, 4]


def included():
    qc = pd.read_csv(OUT / "qc_table.csv")
    return qc[qc["auto_ok"]][["subj", "side"]].values.tolist()


def bout_slope(bout_sig, fs):
    """Linear slope of MDF and MNF within a chewing bout (Hz/min)."""
    t, mdf, mnf = mdf_mnf_timecourse(bout_sig, fs)
    ok_mdf = np.isfinite(mdf)
    ok_mnf = np.isfinite(mnf)
    s_mdf = linregress(t[ok_mdf], mdf[ok_mdf]).slope * 60 if ok_mdf.sum() > 5 else np.nan
    s_mnf = linregress(t[ok_mnf], mnf[ok_mnf]).slope * 60 if ok_mnf.sum() > 5 else np.nan
    return s_mdf, s_mnf


def collect_slopes(sel, fs=1024.0):
    """
    Returns a DataFrame with columns [subj, cond, bout, mdf_slope, mnf_slope].
    """
    rows = []
    for subj, side in sel:
        d = np.load(DERIV / f"{subj}_emg.npz")
        for cond in ("ANE", "PLA"):
            sig = d[f"{cond}_{side}"]
            bouts = reshape_bouts(sig, fs, n_bouts=4)
            for bi, b in enumerate(bouts, start=1):
                sm, sn = bout_slope(b, fs)
                rows.append(dict(subj=subj, cond=cond, bout=bi,
                                 mdf_slope=sm, mnf_slope=sn))
    return pd.DataFrame(rows)


def draw_panel(ax, df, metric, ylabel, panel_label):
    """Line plot of mean ± SEM per bout for ANE and PLA conditions."""
    xs = np.array(BOUTS)
    for cond, color, label in [("ANE", ANE_C, "Anesthesia"),
                                ("PLA", PLA_C, "Placebo")]:
        sub = df[df.cond == cond]
        means, sems = [], []
        for b in BOUTS:
            vals = sub[sub.bout == b][metric].dropna().values
            means.append(vals.mean())
            sems.append(vals.std() / np.sqrt(len(vals)))
        means, sems = np.array(means), np.array(sems)
        ax.plot(xs, means, color=color, lw=1.8, marker="o",
                markersize=5, label=label)
        ax.fill_between(xs, means - sems, means + sems,
                        color=color, alpha=0.18)

    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_xticks(BOUTS)
    ax.set_xticklabels([f"Bout {b}" for b in BOUTS], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.text(-0.13, 1.04, panel_label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

    # N in corner
    n = df.subj.nunique()
    ax.text(0.98, 0.97, f"N = {n}", transform=ax.transAxes,
            fontsize=7.5, ha="right", va="top", color="0.4")


def main():
    sel = included()
    print(f"Included subjects: {len(sel)}")

    df = collect_slopes(sel)
    print("Mean MDF slopes (Hz/min):")
    print(df.groupby(["cond", "bout"])["mdf_slope"].mean().unstack("bout").round(1))

    for metric, ylabel, stem in [
        ("mdf_slope", "MDF slope (Hz/min)", "s5_mdf_slope"),
        ("mnf_slope", "MNF slope (Hz/min)", "s5_mnf_slope"),
    ]:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        draw_panel(ax, df, metric, ylabel, "")
        ax.legend(loc="upper right")
        ax.set_xlabel("Chewing bout")
        fig.tight_layout()
        save_fig(fig, stem, SUPPL)

    # Descriptive statistics summary
    print("\nSummary by condition and bout:")
    print(df.groupby(["cond", "bout"])[["mdf_slope", "mnf_slope"]]
          .agg(["mean", "std"]).round(2))


if __name__ == "__main__":
    main()
