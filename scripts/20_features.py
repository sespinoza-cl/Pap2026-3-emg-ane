"""Calcula features EMG para cada sujeto x condicion x lado, desde los .npz.

Genera:
  - features_all.csv : una fila por (subj, cond, side) con todas las metricas + SNR.
"""
import numpy as np
import pandas as pd
from config import DERIV, OUT, SUBJECTS_36
from emg_features import all_features
from emg_tf import fatigue_slope


def rms(x):
    return float(np.sqrt(np.mean(x**2))) if x.size else np.nan


def main():
    rows = []
    for subj in SUBJECTS_36:
        p = DERIV / f"{subj}_emg.npz"
        if not p.exists():
            print(f"falta npz: {subj}")
            continue
        d = np.load(p, allow_pickle=True)
        fs = float(d["fs"])
        rama = str(d["rama"])
        for side in ("L", "R"):
            rs = d[f"RS_{side}"]
            for cond in ("ANE", "PLA"):
                sig = d[f"{cond}_{side}"]
                feats = all_features(sig, fs, rs_sig=rs)
                fat = fatigue_slope(sig, fs)
                feats["mdf_slope"] = fat["mdf_slope"]
                feats["mnf_slope"] = fat["mnf_slope"]
                snr = (10*np.log10(rms(sig)**2 / rms(rs)**2)
                       if sig.size and rs.size else np.nan)
                rows.append(dict(subj=subj, rama=f"R{rama}", side=side, cond=cond,
                                 fs=fs, dur_s=round(sig.size/fs, 1),
                                 snr_db=round(snr, 2) if snr == snr else np.nan,
                                 **feats))
        print(f"feat {subj} ok")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "features_all.csv", index=False)
    print("\nGuardado:", OUT / "features_all.csv", " filas:", len(df))
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
