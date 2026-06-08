"""Preprocesamiento EMG desde raw (1024 Hz) - Paper3.

Orden correcto (anti-aliasing): se filtra en la frecuencia ORIGINAL (1024 Hz)
y NO se remuestrea, preservando la banda EMG completa (20-450 Hz) para una
estimacion fiel de la frecuencia mediana.

Pasos por sujeto:
  1. Cargar EMG crudo (EXG5-8) en V.
  2. Bandpass 20-450 Hz Butterworth orden 4 (filtfilt, fase cero).
  3. Extraer segmentos monopolares: RS (reposo), ANE (anestesia), PLA (placebo).
  4. Eliminar ruido de linea con ZAPLINE-PLUS (no notch: preserva el espectro;
     mismo metodo que el paper original).
  5. Bipolar desde los monopolares limpios: Left=EXG5-EXG6, Right=EXG7-EXG8.
  6. Guardar .npz por sujeto + tabla resumen con SNR por canal.
"""
import contextlib
import io
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from pyzaplineplus import zapline_plus

from config import (FS_ORIG, BP_LOW, BP_HIGH, BP_ORDER,
                    CHEW_CODE_ON, RS_CODES, BOUT_DUR_S, RS_WIN,
                    bout_condition_map, SUBJECTS_36, DERIV, OUT)
from emg_io import load_emg

LINE_NOISEFREQS = [50, 100, 150, 200]   # fundamental + armonicos (mains 50 Hz)


def bandpass(data, fs):
    nyq = fs / 2.0
    b, a = butter(BP_ORDER, [BP_LOW/nyq, BP_HIGH/nyq], btype="bandpass")
    return filtfilt(b, a, data, axis=1)


def zapline_clean(seg, fs):
    """Zapline-plus sobre 4 canales monopolares (seg: 4 x n). Devuelve 4 x n."""
    if seg.shape[1] < int(2 * fs):
        return seg
    nf = [f for f in LINE_NOISEFREQS if f < fs/2 - 5]
    with contextlib.redirect_stdout(io.StringIO()):
        clean = zapline_plus(seg.T.copy(), fs, noisefreqs=nf,
                             maxfreq=max(nf) + 10, plotResults=False)[0]
    return np.asarray(clean).T


def rms(x):
    return float(np.sqrt(np.mean(x**2))) if x.size else np.nan


def extract_mono_segments(mono, events, fs, rama):
    """Extrae segmentos monopolares (4 x n) por condicion, sin limpiar aun."""
    n = mono.shape[1]
    onsets = np.sort(events[events[:, 1] == CHEW_CODE_ON][:, 0])
    cmap = bout_condition_map(rama)
    bout_len = int(round(BOUT_DUR_S * fs))
    seg = {"ANE": [], "PLA": []}
    used = {"ANE": 0, "PLA": 0}
    for i, on in enumerate(onsets[:8]):
        cond = cmap.get(i)
        if cond is None:
            continue
        a, b = int(on), min(int(on) + bout_len, n)
        if b - a < 0.5 * bout_len:
            continue
        seg[cond].append(mono[:, a:b])
        used[cond] += 1
    rs_on = np.sort(events[np.isin(events[:, 1], RS_CODES)][:, 0])
    w0, w1 = int(RS_WIN[0]*fs), int(RS_WIN[1]*fs)
    rs = []
    for on in rs_on:
        a, b = int(on) + w0, min(int(on) + w1, n)
        if b - a >= 5*fs:
            rs.append(mono[:, a:b])
    out = {"used": used}
    out["ANE_M"] = np.concatenate(seg["ANE"], axis=1) if seg["ANE"] else np.zeros((4, 0))
    out["PLA_M"] = np.concatenate(seg["PLA"], axis=1) if seg["PLA"] else np.zeros((4, 0))
    out["RS_M"] = np.concatenate(rs, axis=1) if rs else np.zeros((4, 0))
    return out


def main():
    rows = []
    for subj in SUBJECTS_36:
        try:
            d = load_emg(subj)
            fs = d["sfreq"]
            mono = bandpass(d["data"], fs)
            seg = extract_mono_segments(mono, d["events"], fs, d["rama"])

            # Zapline-plus por segmento y bipolar desde monopolares limpios
            out = {}
            for cond in ("ANE", "PLA", "RS"):
                m = zapline_clean(seg[f"{cond}_M"], fs)
                out[f"{cond}_M"] = m
                out[f"{cond}_L"] = (m[0] - m[1]) if m.shape[1] else np.array([])
                out[f"{cond}_R"] = (m[2] - m[3]) if m.shape[1] else np.array([])

            def snr(cond, side):
                xs, rs = out[f"{cond}_{side}"], out[f"RS_{side}"]
                if xs.size == 0 or rs.size == 0:
                    return np.nan
                return 10*np.log10(rms(xs)**2 / rms(rs)**2)

            np.savez_compressed(
                DERIV / f"{subj}_emg.npz",
                ANE_L=out["ANE_L"], ANE_R=out["ANE_R"],
                PLA_L=out["PLA_L"], PLA_R=out["PLA_R"],
                RS_L=out["RS_L"], RS_R=out["RS_R"],
                ANE_M=out["ANE_M"], PLA_M=out["PLA_M"], RS_M=out["RS_M"],
                fs=fs, rama=d["rama"])

            rows.append(dict(
                subj=subj, file=d["file"], rama=f"R{d['rama']}", fs=fs,
                bouts_ANE=seg["used"]["ANE"], bouts_PLA=seg["used"]["PLA"],
                dur_ANE_s=round(out["ANE_L"].size/fs, 1),
                dur_PLA_s=round(out["PLA_L"].size/fs, 1),
                dur_RS_s=round(out["RS_L"].size/fs, 1),
                SNR_ANE_L=round(snr("ANE", "L"), 2), SNR_ANE_R=round(snr("ANE", "R"), 2),
                SNR_PLA_L=round(snr("PLA", "L"), 2), SNR_PLA_R=round(snr("PLA", "R"), 2)))
            print(f"OK  {subj:5} R{d['rama']} bouts(A/P)={seg['used']['ANE']}/{seg['used']['PLA']} "
                  f"SNR_R A/P={rows[-1]['SNR_ANE_R']}/{rows[-1]['SNR_PLA_R']} dB", flush=True)
        except Exception as e:
            print(f"ERR {subj:5}: {type(e).__name__}: {e}", flush=True)
            rows.append(dict(subj=subj, file="ERROR", rama="", fs=np.nan))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "preprocess_summary.csv", index=False)
    print("\nGuardado:", OUT / "preprocess_summary.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
