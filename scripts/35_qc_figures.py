"""QC montage — S1 supplementary figure.

6×6 grid showing 10 s of rectified EMG + RMS envelope for all 36 subjects
(Anesthesia condition, physiologically selected side).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import OUT, DERIV, SUBJECTS_36
from emg_features import rms_envelope
from fig_style import ANEC, SUPPL, save_fig

plt.rcParams.update({"figure.dpi": 120, "font.size": 7})


def main():
    qc  = pd.read_csv(OUT / "qc_table.csv").set_index("subj")
    fs  = 1024.0
    ns  = int(10 * fs)

    fig, axes = plt.subplots(6, 6, figsize=(16, 11))
    for ax, subj in zip(axes.ravel(), SUBJECTS_36):
        p = DERIV / f"{subj}_emg.npz"
        if not p.exists():
            ax.axis("off"); continue
        d    = np.load(p)
        side = qc.loc[subj, "side"]
        ok   = qc.loc[subj, "auto_ok"]
        snr  = qc.loc[subj, "snr_min"]
        sig  = d[f"ANE_{side}"][:ns]
        t    = np.arange(len(sig)) / fs
        rect = np.abs(sig)
        env  = rms_envelope(rect, fs)
        ax.plot(t, rect * 1e3, color="0.5", lw=0.4)
        ax.plot(t, env  * 1e3, color=ANEC,  lw=1.0)
        col = "black" if ok else "red"
        ax.set_title(f"{subj} [{side}] SNR={snr:.0f}" + ("" if ok else " !"),
                     color=col, fontsize=6.5)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    save_fig(fig, "s1_qc_montaje", SUPPL)


if __name__ == "__main__":
    main()
