"""Analisis tiempo-frecuencia del EMG masticatorio.

  - Espectrograma (STFT) y curso temporal de MDF/MNF.
  - Pendiente de fatiga: cambio de MDF/MNF dentro del bout de 60 s
    (promediando los 4 bouts de la condicion). MDF decreciente = fatiga /
    enlentecimiento de la velocidad de conduccion.
  - Espectrograma promedio por condicion (para figuras de grupo).
"""
import numpy as np
from scipy.signal import spectrogram
from scipy.stats import linregress

BOUT_S = 60.0


def reshape_bouts(sig, fs, bout_s=BOUT_S, n_bouts=4):
    """Si la senal son exactamente n_bouts x bout_s, la separa en bouts."""
    L = int(round(bout_s * fs))
    if sig.size == n_bouts * L:
        return sig.reshape(n_bouts, L)
    # fallback: cuantos bouts completos quepan
    nb = sig.size // L
    return sig[:nb * L].reshape(nb, L) if nb >= 1 else sig[None, :]


def spectro(sig, fs, win_s=0.5, overlap=0.5, fmin=20, fmax=450):
    nper = int(win_s * fs)
    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=nper,
                            noverlap=int(nper * overlap), scaling="density")
    m = (f >= fmin) & (f <= fmax)
    return f[m], t, Sxx[m, :]


def mdf_mnf_timecourse(sig, fs, win_s=0.5, fmin=20, fmax=450):
    f, t, Sxx = spectro(sig, fs, win_s=win_s, fmin=fmin, fmax=fmax)
    tot = np.trapz(Sxx, f, axis=0)
    mnf = np.trapz(f[:, None] * Sxx, f, axis=0) / np.where(tot > 0, tot, np.nan)
    # MDF por columna
    cum = np.cumsum((Sxx[:-1, :] + Sxx[1:, :]) / 2 * np.diff(f)[:, None], axis=0)
    mdf = np.full(t.shape, np.nan)
    for j in range(Sxx.shape[1]):
        if tot[j] > 0:
            idx = np.searchsorted(cum[:, j], tot[j] / 2)
            mdf[j] = f[min(idx + 1, len(f) - 1)]
    return t, mdf, mnf


def fatigue_slope(sig, fs, win_s=0.5):
    """Pendiente (Hz/min) de MDF y MNF dentro del bout, promediando bouts."""
    bouts = reshape_bouts(sig, fs)
    mdf_stack, mnf_stack, tref = [], [], None
    for b in bouts:
        t, mdf, mnf = mdf_mnf_timecourse(b, fs, win_s=win_s)
        if tref is None:
            tref = t
        if len(t) == len(tref):
            mdf_stack.append(mdf); mnf_stack.append(mnf)
    if not mdf_stack:
        return dict(mdf_slope=np.nan, mnf_slope=np.nan,
                    mdf_t=np.array([]), mdf_mean=np.array([]))
    mdf_mean = np.nanmean(np.vstack(mdf_stack), axis=0)
    mnf_mean = np.nanmean(np.vstack(mnf_stack), axis=0)
    ok = np.isfinite(mdf_mean)
    mdf_slope = linregress(tref[ok], mdf_mean[ok]).slope * 60 if ok.sum() > 2 else np.nan
    okn = np.isfinite(mnf_mean)
    mnf_slope = linregress(tref[okn], mnf_mean[okn]).slope * 60 if okn.sum() > 2 else np.nan
    return dict(mdf_slope=float(mdf_slope), mnf_slope=float(mnf_slope),
                mdf_t=tref, mdf_mean=mdf_mean, mnf_mean=mnf_mean)


def mean_spectrogram(sig, fs, win_s=0.5, fmin=20, fmax=450):
    """Espectrograma promedio en el tiempo-dentro-de-bout (0-60 s)."""
    bouts = reshape_bouts(sig, fs)
    stack, fref, tref = [], None, None
    for b in bouts:
        f, t, Sxx = spectro(b, fs, win_s=win_s, fmin=fmin, fmax=fmax)
        if fref is None:
            fref, tref = f, t
        if Sxx.shape == (len(fref), len(tref)):
            stack.append(Sxx)
    if not stack:
        return fref, tref, None
    return fref, tref, np.mean(np.stack(stack), axis=0)
