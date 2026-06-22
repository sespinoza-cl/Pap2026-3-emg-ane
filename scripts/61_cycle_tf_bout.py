"""TF cycle-locked resuelto por BOUT: compara la diferencia ANE-PLA en el
bout 1 (anestesia maxima) vs bout 4 (washout). El efecto deberia verse en bout 1.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_array_morlet
from config import OUT, DERIV
from emg_cycles import detect_cycle_peaks, epoch_around_peaks
from emg_tf import reshape_bouts
from fig_style import RC, FIGSIZE_WIDE, SUPPL, save_fig

FREQS = np.logspace(np.log10(20), np.log10(200), 35)  # BioSemi CIC: -3dB ~205 Hz
N_CYCLES = np.clip(FREQS / 6.0, 4, 15)


def bout_tf(bout_sig, fs, rs):
    pks = detect_cycle_peaks(bout_sig, fs, rs_sig=rs)
    ep, t = epoch_around_peaks(bout_sig, fs, pks, -0.4, 0.4)
    if ep.shape[0] < 5:
        return None, t
    power = tfr_array_morlet(ep[:, None, :], sfreq=fs, freqs=FREQS,
                             n_cycles=N_CYCLES, output="power", verbose="ERROR")
    return 10 * np.log10(power[:, 0].mean(axis=0)), t


def main():
    cache = DERIV / "cycle_tf_bout.npz"
    if cache.exists():
        print("Loading from cache:", cache)
        d    = np.load(cache)
        tref = d["t"]
        M    = {(b, c): d[f"{b}_{c}"] for b in (1, 4) for c in ("ANE", "PLA")}
    else:
        qc  = pd.read_csv(OUT / "qc_table.csv")
        inc = qc[qc["auto_ok"]]
        store = {(b, c): [] for b in (1, 4) for c in ("ANE", "PLA")}
        tref  = None
        for _, q in inc.iterrows():
            subj, side = q["subj"], q["side"]
            d  = np.load(DERIV / f"{subj}_emg.npz")
            fs = float(d["fs"]); rs = d[f"RS_{side}"]
            for cond in ("ANE", "PLA"):
                bouts = reshape_bouts(d[f"{cond}_{side}"], fs, n_bouts=4)
                for bi in (1, 4):
                    p, t = bout_tf(bouts[bi - 1], fs, rs)
                    if p is not None:
                        store[(bi, cond)].append(p); tref = t
            print(f"tf-bout {subj} ok", flush=True)
        M = {k: np.mean(np.stack(v), axis=0) for k, v in store.items()}
        np.savez_compressed(cache, freqs=FREQS, t=tref,
                            **{f"{b}_{c}": M[(b, c)] for b in (1, 4) for c in ("ANE", "PLA")})

    plt.rcParams.update(RC)
    ext  = [tref[0], tref[-1], FREQS[0], FREQS[-1]]
    D1   = M[(1, "ANE")] - M[(1, "PLA")]
    D4   = M[(4, "ANE")] - M[(4, "PLA")]
    vlim = np.nanpercentile(np.abs(np.concatenate([D1, D4])), 99)

    for D, stem in [(D1, "s6_tf_bout1_diff"), (D4, "s6_tf_bout4_diff")]:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        im = ax.imshow(D, aspect="auto", origin="lower", extent=ext,
                       cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("Time relative to cycle peak (s)")
        ax.set_ylabel("Frequency (Hz)")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("dB (ANE − PLA)", fontsize=9)
        fig.tight_layout()
        save_fig(fig, stem, SUPPL)


if __name__ == "__main__":
    main()
