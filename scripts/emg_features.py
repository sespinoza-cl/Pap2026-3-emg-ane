"""Extraccion de caracteristicas EMG masticatorio.

Dos familias:
  - Espectrales (Welch sobre la senal bipolar continua): MDF, MNF, potencia
    total, frecuencia pico, potencia por sub-banda. (MDF/MNF = reclutamiento y
    velocidad de conduccion; angulo novedoso vs el analisis previo del lab.)
  - Envolvente / ciclos masticatorios: tasa de ciclos, duracion, regularidad
    (CV del intervalo inter-pico), amplitud, area, duty cycle.
"""
import numpy as np
from scipy.signal import welch, find_peaks


# ----------------------- Espectrales -----------------------
def spectral_features(sig, fs, fmin=20.0, fmax=450.0):
    if sig.size < fs:  # < 1 s
        return {k: np.nan for k in
                ["MDF", "MNF", "peakF", "totpow", "p_20_60", "p_60_150",
                 "p_150_250", "p_250_450"]}
    nperseg = int(min(len(sig), fs))  # ventana 1 s
    f, pxx = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    m = (f >= fmin) & (f <= fmax)
    f, pxx = f[m], pxx[m]
    tot = np.trapz(pxx, f)
    cum = np.cumsum((pxx[:-1] + pxx[1:]) / 2 * np.diff(f))
    mdf = f[np.searchsorted(cum, tot / 2) + 1] if tot > 0 else np.nan
    mnf = np.trapz(f * pxx, f) / tot if tot > 0 else np.nan
    peakf = f[np.argmax(pxx)]

    def bp(lo, hi):
        mm = (f >= lo) & (f < hi)
        return float(np.trapz(pxx[mm], f[mm]))
    return dict(MDF=float(mdf), MNF=float(mnf), peakF=float(peakf),
                totpow=float(tot), p_20_60=bp(20, 60), p_60_150=bp(60, 150),
                p_150_250=bp(150, 250), p_250_450=bp(250, 450))


# ----------------------- Envolvente / ciclos -----------------------
def rms_envelope(sig, fs, win_ms=100):
    w = max(1, int(round(win_ms * fs / 1000)))
    return np.sqrt(np.convolve(sig**2, np.ones(w)/w, mode="same"))


def cycle_features(sig, fs, rs_sig=None, min_dist_s=0.4, max_ipi_s=1.5):
    """Detecta ciclos masticatorios sobre la envolvente RMS normalizada por reposo."""
    env = rms_envelope(np.abs(sig), fs)
    if rs_sig is not None and rs_sig.size > fs:
        rs_env = rms_envelope(np.abs(rs_sig), fs)
        mu, sd = rs_env.mean(), rs_env.std() + 1e-12
    else:
        mu, sd = env.mean(), env.std() + 1e-12
    z = (env - mu) / sd
    thr = np.percentile(z, 50)
    pks, _ = find_peaks(z, height=thr, distance=int(min_dist_s * fs))
    if len(pks) < 3:
        return dict(n_cycles=len(pks), chew_rate=np.nan, cv_ipi=np.nan,
                    med_dur=np.nan, med_amp=np.nan, med_area=np.nan, duty=np.nan)
    ipi = np.diff(pks) / fs
    ipi = ipi[ipi <= max_ipi_s]          # descartar pausas
    chew_rate = 1.0 / np.median(ipi) if ipi.size else np.nan
    cv_ipi = np.std(ipi) / np.mean(ipi) if ipi.size else np.nan

    # ciclos: onset(min antes del pico) - pico - offset(min hasta el siguiente)
    durs, amps, areas = [], [], []
    rect = np.abs(sig)
    back = int(0.3 * fs)
    for k in range(len(pks) - 1):
        pk, nxt = pks[k], pks[k+1]
        a = max(0, pk - back)
        onset = a + np.argmin(z[a:pk+1]) if pk > a else a
        offset = pk + np.argmin(z[pk:nxt+1])
        if not (onset < pk < offset):
            continue
        seg = rect[onset:offset]
        durs.append((offset - onset)/fs)
        amps.append(rect[pk])
        areas.append(seg.sum()/fs)
    dur_total = len(sig)/fs
    return dict(n_cycles=len(pks),
                chew_rate=float(chew_rate), cv_ipi=float(cv_ipi),
                med_dur=float(np.median(durs)) if durs else np.nan,
                med_amp=float(np.median(amps)) if amps else np.nan,
                med_area=float(np.median(areas)) if areas else np.nan,
                duty=float(np.sum(durs)/dur_total) if durs else np.nan)


def all_features(sig, fs, rs_sig=None):
    f = spectral_features(sig, fs)
    f.update(cycle_features(sig, fs, rs_sig=rs_sig))
    f["rms_amp"] = float(np.sqrt(np.mean(sig**2))) if sig.size else np.nan
    return f
