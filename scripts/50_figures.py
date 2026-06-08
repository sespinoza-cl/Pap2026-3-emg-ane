"""Supplementary figures: forest plot (S2), grand-average PSD (S3),
MDF fatigue time-course (S4).

Outputs → overleaf/figures/suppl/
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from config import OUT, DERIV
from emg_tf import mean_spectrogram, mdf_mnf_timecourse, reshape_bouts
from fig_style import RC, FIGSIZE, FIGSIZE_WIDE, ANEC, PLAC, SUPPL, save_fig

_FEAT_LABELS = {
    "MDF":        "Median frequency (MDF)",
    "MNF":        "Mean frequency (MNF)",
    "peakF":      "Peak frequency",
    "totpow":     "Total power",
    "rms_amp":    "RMS amplitude",
    "p_20_60":    "Power 20–60 Hz",
    "p_60_150":   "Power 60–150 Hz",
    "p_150_250":  "Power 150–250 Hz",
    "p_250_450":  "Power 250–450 Hz",
    "chew_rate":  "Chewing rate",
    "cv_ipi":     "CV of IPI",
    "med_dur":    "Median cycle duration",
    "med_amp":    "Median cycle amplitude",
    "med_area":   "Median cycle area",
    "duty":       "Duty cycle",
    "mdf_slope":  "MDF slope",
    "mnf_slope":  "MNF slope",
}


def included():
    qc = pd.read_csv(OUT / "qc_table.csv")
    return qc[qc["auto_ok"]][["subj", "side"]].values.tolist()


# ── S2 — Forest plot of effect sizes + equivalence ────────────────────────────

def fig_forest():
    plt.rcParams.update(RC)
    res = pd.read_csv(OUT / "stats_ANE_vs_PLA.csv").iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    y  = np.arange(len(res))
    se = 1 / np.sqrt(res["n"])
    colors = [ANEC if v >= 0 else PLAC for v in res["cohen_dz"]]
    for i, (dz, err, c) in enumerate(zip(res["cohen_dz"], 1.96 * se, colors)):
        ax.errorbar(dz, i, xerr=err, fmt="o", color=c, capsize=3,
                    ms=5, elinewidth=0.9, capthick=0.9)
    ax.axvspan(-0.5, 0.5, color="grey", alpha=0.15)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_FEAT_LABELS.get(f, f) for f in res["feature"]], fontsize=8.5)
    ax.set_xlabel("Cohen's $d_z$  (Anesthesia $-$ Placebo)")
    fig.tight_layout()
    save_fig(fig, "s2_forest", SUPPL)


# ── S3 — Grand-average PSD ───────────────────────────────────────────────────

def fig_psd():
    plt.rcParams.update(RC)
    sel  = included()
    fs   = 1024.0
    psd_a, psd_p, fref = [], [], None
    for subj, side in sel:
        d = np.load(DERIV / f"{subj}_emg.npz")
        for store, cond in [(psd_a, "ANE"), (psd_p, "PLA")]:
            sig = d[f"{cond}_{side}"]
            if sig.size < fs:
                continue
            f, pxx = welch(sig, fs=fs, nperseg=int(fs))
            m = (f >= 20) & (f <= 450)
            fref = f[m]
            store.append(pxx[m] / np.trapz(pxx[m], f[m]))
    A = np.vstack(psd_a); P = np.vstack(psd_p)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for arr, c, lab in [(A, ANEC, "Anesthesia"), (P, PLAC, "Placebo")]:
        mu  = arr.mean(0)
        sem = arr.std(0) / np.sqrt(len(arr))
        ax.plot(fref, mu, color=c, label=lab)
        ax.fill_between(fref, mu - sem, mu + sem, color=c, alpha=0.25)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "s3_psd", SUPPL)


# ── S4 — MDF time course within bout (fatigue) ────────────────────────────────

def fig_fatigue():
    plt.rcParams.update(RC)
    sel  = included()
    fs   = 1024.0
    mdf_a, mdf_p, tref = [], [], None
    for subj, side in sel:
        d = np.load(DERIV / f"{subj}_emg.npz")
        for store, cond in [(mdf_a, "ANE"), (mdf_p, "PLA")]:
            sig   = d[f"{cond}_{side}"]
            bouts = reshape_bouts(sig, fs)
            ms    = []
            for b in bouts:
                t, mdf, _ = mdf_mnf_timecourse(b, fs)
                if tref is None:
                    tref = t
                if len(t) == len(tref):
                    ms.append(mdf)
            if ms:
                store.append(np.nanmean(np.vstack(ms), 0))
    A = np.vstack(mdf_a); P = np.vstack(mdf_p)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for arr, c, lab in [(A, ANEC, "Anesthesia"), (P, PLAC, "Placebo")]:
        mu  = np.nanmean(arr, 0)
        sem = np.nanstd(arr, 0) / np.sqrt(len(arr))
        ax.plot(tref, mu, color=c, label=lab)
        ax.fill_between(tref, mu - sem, mu + sem, color=c, alpha=0.25)
    ax.set_xlabel("Time within bout (s)")
    ax.set_ylabel("Median frequency (Hz)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "s4_fatigue", SUPPL)


if __name__ == "__main__":
    fig_forest(); print("S2 ok")
    fig_psd();    print("S3 ok")
    fig_fatigue(); print("S4 ok")
