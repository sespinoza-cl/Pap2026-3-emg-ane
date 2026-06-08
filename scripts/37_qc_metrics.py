"""Metricas objetivas de calidad de senal EMG por sujeto (lado elegido, ANE).

  - burst_ratio   : mediana envolvente chew / mediana envolvente reposo (>~3 bueno)
  - pct_active    : % del tiempo de chew con envolvente > 2x mediana reposo
  - env_peak_hz   : frecuencia del pico del espectro de la ENVOLVENTE (ritmicidad
                    masticatoria; debe caer ~1-2 Hz)
  - env_peak_prom : prominencia relativa de ese pico (ritmicidad; mas alto=mas ritmico)
  - kurtosis      : kurtosis de la senal (muy baja => posible saturacion/clipping)
  - sat_pct       : % de muestras cerca del maximo absoluto (clipping)
"""
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis
from config import OUT, DERIV
from emg_features import rms_envelope


def metrics(sig, rs, fs):
    env = rms_envelope(np.abs(sig), fs)
    rs_env = rms_envelope(np.abs(rs), fs)
    med_rs = np.median(rs_env) + 1e-15
    burst_ratio = np.median(env) / med_rs
    pct_active = 100 * np.mean(env > 2 * med_rs)
    # ritmicidad: PSD de la envolvente
    f, p = welch(env - env.mean(), fs=fs, nperseg=int(min(len(env), 8*fs)))
    band = (f >= 0.5) & (f <= 3.0)
    if band.any():
        ip = np.argmax(p[band])
        env_peak_hz = f[band][ip]
        env_peak_prom = p[band][ip] / (np.median(p[(f > 0.1) & (f <= 6)]) + 1e-30)
    else:
        env_peak_hz, env_peak_prom = np.nan, np.nan
    k = kurtosis(sig)
    amax = np.max(np.abs(sig))
    sat_pct = 100 * np.mean(np.abs(sig) > 0.999 * amax)
    return dict(burst_ratio=burst_ratio, pct_active=pct_active,
                env_peak_hz=env_peak_hz, env_peak_prom=env_peak_prom,
                kurtosis=k, sat_pct=sat_pct)


def main():
    qc = pd.read_csv(OUT / "qc_table.csv").set_index("subj")
    rows = []
    for subj in qc.index:
        d = np.load(DERIV / f"{subj}_emg.npz")
        fs = float(d["fs"]); side = qc.loc[subj, "side"]
        m = metrics(d[f"ANE_{side}"], d[f"RS_{side}"], fs)
        rows.append(dict(subj=subj, side=side, snr_min=qc.loc[subj, "snr_min"], **m))
    df = pd.DataFrame(rows)
    # z-scores robustos para flag de atipicos (sobre los auto-OK)
    for col in ["burst_ratio", "env_peak_prom", "kurtosis"]:
        med = df[col].median(); mad = (df[col] - med).abs().median() + 1e-12
        df[col + "_z"] = 0.6745 * (df[col] - med) / mad
    df = df.sort_values("env_peak_prom")
    df.to_csv(OUT / "qc_metrics.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    cols = ["subj", "side", "snr_min", "burst_ratio", "pct_active",
            "env_peak_hz", "env_peak_prom", "kurtosis", "sat_pct"]
    print(df[cols].round(2).to_string(index=False))
    print("\n>>> PS6 / S7 vs mediana del grupo:")
    print(df[df.subj.isin(["PS6", "S7"])][cols + ["burst_ratio_z", "env_peak_prom_z"]].round(2).to_string(index=False))
    print("\nMedianas grupo: burst_ratio=%.2f pct_active=%.1f env_peak_prom=%.1f kurtosis=%.2f" %
          (df["burst_ratio"].median(), df["pct_active"].median(), df["env_peak_prom"].median(), df["kurtosis"].median()))


if __name__ == "__main__":
    main()
