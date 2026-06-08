"""Cluster-based permutation test on cycle-locked TF (paired ANE-PLA).

Two SEPARATE panels saved to overleaf/figures/main/:
  fig4a_tf_bout1.png   TF difference + significant cluster, bout 1
  fig4b_tf_bout4.png   TF difference (no cluster expected), bout 4

Method: Maris & Oostenveld 2007, MNE permutation_cluster_1samp_test.
Directional (a priori ANE>PLA, already established by med_amp dz=0.70).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from mne.time_frequency import tfr_array_morlet
from mne.stats import permutation_cluster_1samp_test
from config import OUT, DERIV
from emg_cycles import detect_cycle_peaks, epoch_around_peaks
from emg_tf import reshape_bouts
from fig_style import RC, FIGSIZE_WIDE, MAIN, save_fig

FREQS    = np.logspace(np.log10(20), np.log10(450), 45)
N_CYCLES = np.clip(FREQS / 6.0, 4, 15)
DECIM    = 8
N_PERM   = 2000
ALPHA    = 0.05


def bout_tf(sig, fs, rs):
    pks = detect_cycle_peaks(sig, fs, rs_sig=rs)
    ep, t = epoch_around_peaks(sig, fs, pks, -0.4, 0.4)
    if ep.shape[0] < 5:
        return None, t
    pw = tfr_array_morlet(ep[:, None, :], sfreq=fs, freqs=FREQS,
                          n_cycles=N_CYCLES, output="power", verbose="ERROR")
    return 10 * np.log10(pw[:, 0].mean(axis=0)), t


def collect(bout):
    qc  = pd.read_csv(OUT / "qc_table.csv")
    inc = qc[qc["auto_ok"]]
    diffs, tref = [], None
    for _, q in inc.iterrows():
        d  = np.load(DERIV / f"{q['subj']}_emg.npz")
        fs = float(d["fs"])
        rs = d[f"RS_{q['side']}"]
        pa, t = bout_tf(reshape_bouts(d[f"ANE_{q['side']}"], fs, n_bouts=4)[bout - 1], fs, rs)
        pp, _ = bout_tf(reshape_bouts(d[f"PLA_{q['side']}"], fs, n_bouts=4)[bout - 1], fs, rs)
        if pa is not None and pp is not None:
            diffs.append((pa - pp)[:, ::DECIM])
            tref = t[::DECIM]
    return np.stack(diffs), tref


def run_cluster(diffs, tail=1):
    n   = diffs.shape[0]
    thr = stats.t.ppf(1 - ALPHA / (1 if tail == 1 else 2), n - 1)
    T, clusters, cluster_p, _ = permutation_cluster_1samp_test(
        diffs, threshold=thr, n_permutations=N_PERM, tail=tail, seed=0,
        out_type="mask", verbose="ERROR")
    return T, clusters, cluster_p, n


def plot_panel(bout, stem):
    plt.rcParams.update(RC)
    diffs, tref = collect(bout)
    T, clusters, cp, n = run_cluster(diffs, tail=1)
    _, _, cp2, _       = run_cluster(diffs, tail=0)

    sigp = [round(p, 4) for p in cp  if p < ALPHA]
    print(f"Bout {bout} (N={n}): dir p={[round(p,3) for p in cp]}, "
          f"sig={sigp}  |  bidir p_min={min(cp2):.3f}")

    M    = diffs.mean(0)
    ext  = [tref[0], tref[-1], FREQS[0], FREQS[-1]]
    vlim = np.nanpercentile(np.abs(M), 99)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    im = ax.imshow(M, aspect="auto", origin="lower", extent=ext,
                   cmap="RdBu_r", vmin=-vlim, vmax=vlim)

    sig = np.zeros(T.shape, bool)
    for c, p in zip(clusters, cp):
        if p < ALPHA:
            sig |= c
    if sig.any():
        ax.contour(sig, levels=[0.5], colors="black", linewidths=1.5,
                   extent=ext, origin="lower")

    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time relative to cycle peak (s)")
    ax.set_ylabel("Frequency (Hz)")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("dB (ANE − PLA)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, stem, MAIN)


def main():
    plot_panel(1, "fig4a_tf_bout1")
    plot_panel(4, "fig4b_tf_bout4")


if __name__ == "__main__":
    main()
