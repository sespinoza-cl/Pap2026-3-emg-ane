"""Publication-quality per-subject EMG diagnostics for Supplementary Fig. S9.

Generates fig_s9_diagnostics.png: 6 representative subjects (2 high-effect,
1 moderate, 2 borderline-null, 1 excluded) in a 3-column x 2-row layout.
Each subject panel shows bout-1 waveform (ANE vs PLA) and PSD.

Usage: python 36_subject_diag.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch, find_peaks
from config import OUT, DERIV

# ── Aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 7.5,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

C_ANE  = "#E07B39"   # orange — anesthesia
C_PLA  = "#3B7DC4"   # blue   — placebo
C_REST = "#6AAF6A"   # green  — rest

BOUT_DUR_S = 60.0
FS         = 1024.0

# Subjects: (ID, side, label for figure)
SUBJECTS = [
    ("PS6",  "R", "PS6\n(high effect)"),
    ("M6",   "R", "M6\n(high effect)"),
    ("PS7",  "R", "PS7\n(moderate)"),
    ("S4",   "R", "S4\n(borderline)"),
    ("S6",   "R", "S6\n(borderline)"),
    ("M3",   "L", "M3\n(excluded)"),
]


def load_bout1(npz, side):
    """Return ANE, PLA, REST bipolar arrays; bout 1 only (first 60 s)."""
    n = int(BOUT_DUR_S * FS)
    ane = npz[f"ANE_{side}"][:n]
    pla = npz[f"PLA_{side}"][:n]
    rs  = npz[f"RS_{side}"]
    return ane, pla, rs


def detect_peaks(sig):
    """Peaks on rectified signal (µV)."""
    rect = np.abs(sig) * 1e6
    env  = _smooth(rect, int(0.05 * FS))          # 50-ms smoothing
    thr  = np.percentile(env, 30)
    pks, _ = find_peaks(env, height=thr,
                        distance=int(0.3 * FS),
                        prominence=max(15, thr * 0.3))
    return pks, env


def _smooth(x, w):
    if w < 2:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def plot_waveform(ax, ane, pla, fs=FS, t_show=8.0, title=""):
    """Bout-1 rectified waveform, first t_show seconds."""
    n = int(t_show * fs)
    t = np.arange(n) / fs

    for sig, color, lab in [(pla, C_PLA, "Placebo"),
                             (ane, C_ANE, "Anesthesia")]:
        rect = np.abs(sig[:n]) * 1e6            # µV
        env  = _smooth(rect, int(0.05 * fs))
        ax.fill_between(t, rect, alpha=0.18, color=color, linewidth=0)
        ax.plot(t, env, color=color, lw=1.1, label=lab)

    # cycle peaks from anesthesia trace
    pks, env_full = detect_peaks(ane)
    pks_show = pks[pks < n]
    ax.scatter(pks_show / fs,
               np.abs(ane[pks_show]) * 1e6,
               s=12, color=C_ANE, zorder=5, marker="v", alpha=0.7)

    ax.set_xlim(0, t_show)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("Amplitude (µV)", fontsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)


def plot_psd(ax, ane, pla, rs, fs=FS):
    """Power spectral density for the full 60-s bout."""
    nperseg = int(fs)
    for sig, color, lab in [(rs,  C_REST, "Rest"),
                             (pla, C_PLA,  "Placebo"),
                             (ane, C_ANE,  "Anesthesia")]:
        if sig.size < nperseg:
            continue
        f, pxx = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        mask = (f >= 20) & (f <= 450)
        ax.semilogy(f[mask], pxx[mask] * 1e12, color=color, lw=0.9, label=lab)

    ax.set_xlabel("Frequency (Hz)", fontsize=7)
    ax.set_ylabel("PSD (µV²/Hz)", fontsize=7)
    ax.set_xlim(20, 450)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")


def make_s9_figure(out_path):
    n_subj = len(SUBJECTS)
    n_cols = 3
    n_rows = 2   # 2 rows × 3 cols = 6 subjects

    fig = plt.figure(figsize=(13, 7.5))
    outer = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                              wspace=0.38, hspace=0.52)

    for idx, (subj, side, label) in enumerate(SUBJECTS):
        row, col = divmod(idx, n_cols)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row, col],
            hspace=0.45, height_ratios=[1.3, 1])

        npz_path = DERIV / f"{subj}_emg.npz"
        if not npz_path.exists():
            print(f"  WARNING: {npz_path} not found — skipping {subj}")
            continue

        d   = np.load(npz_path)
        ane, pla, rs = load_bout1(d, side)

        ax_wave = fig.add_subplot(inner[0])
        ax_psd  = fig.add_subplot(inner[1])

        # Waveform (bout 1, first 8 s)
        is_excluded = (subj == "M3")
        title_text = f"{label}  |  side {side}"
        plot_waveform(ax_wave, ane, pla, t_show=8.0,
                      title=title_text)
        if is_excluded:
            ax_wave.set_facecolor("#fff0f0")
            ax_wave.text(0.5, 0.92, "EXCLUDED (dead electrode)",
                         transform=ax_wave.transAxes,
                         ha="center", va="top", fontsize=6.5,
                         color="#cc0000", fontstyle="italic")

        # PSD
        plot_psd(ax_psd, ane, pla, rs)

    fig.suptitle(
        "Supplementary Fig. S9 — Per-subject bipolar masseter EMG diagnostics\n"
        "(bout 1 waveform + power spectral density, Anesthesia vs Placebo)",
        fontsize=9, y=1.01
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    out_path = OUT / "fig_s9_diagnostics.png"
    make_s9_figure(out_path)
