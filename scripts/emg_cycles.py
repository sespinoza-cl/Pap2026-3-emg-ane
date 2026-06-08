"""Deteccion de picos de ciclos masticatorios y epocado para tiempo-frecuencia."""
import numpy as np
from emg_features import rms_envelope


def detect_cycle_peaks(sig, fs, rs_sig=None, min_dist_s=0.4, max_ipi_s=1.5):
    """Devuelve indices (en muestras de sig) de los picos de ciclo masticatorio."""
    from scipy.signal import find_peaks
    env = rms_envelope(np.abs(sig), fs)
    if rs_sig is not None and rs_sig.size > fs:
        rs_env = rms_envelope(np.abs(rs_sig), fs)
        mu, sd = rs_env.mean(), rs_env.std() + 1e-12
    else:
        mu, sd = env.mean(), env.std() + 1e-12
    z = (env - mu) / sd
    thr = np.percentile(z, 50)
    pks, _ = find_peaks(z, height=thr, distance=int(min_dist_s * fs))
    # descartar picos cuyo intervalo al vecino sea una pausa (> max_ipi)
    if len(pks) >= 2:
        ipi = np.diff(pks) / fs
        keep = np.ones(len(pks), bool)
        # marcar como pausa los que estan aislados (ambos lados > max_ipi)
        for i in range(len(pks)):
            left = ipi[i-1] if i > 0 else 0
            right = ipi[i] if i < len(ipi) else 0
            if (i > 0 and left > max_ipi_s) and (i < len(ipi) and right > max_ipi_s):
                keep[i] = False
        pks = pks[keep]
    return pks


def epoch_around_peaks(sig, fs, peaks, tmin=-0.4, tmax=0.4):
    """Epoca sig alrededor de cada pico. Devuelve (n_epochs, n_times) y el vector t."""
    a = int(round(tmin * fs)); b = int(round(tmax * fs))
    n = sig.size
    ep = []
    for p in peaks:
        i0, i1 = p + a, p + b
        if i0 >= 0 and i1 <= n:
            ep.append(sig[i0:i1])
    if not ep:
        return np.zeros((0, b - a)), np.arange(a, b) / fs
    L = b - a
    ep = [e[:L] for e in ep if len(e) >= L]
    return np.vstack(ep), np.arange(a, b) / fs
