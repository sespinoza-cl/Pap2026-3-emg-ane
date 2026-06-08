"""Loader universal de EMG crudo (BDF, EEGLAB .set, o .set v7.3) para Exp1.

Devuelve siempre la senal EMG (EXG5-8) en Voltios a 1024 Hz, los eventos
(sample, code) y la rama de counterbalanceo (R1/R2).
"""
import numpy as np
import mne
import h5py
from config import RAW_DIR, EMG_LABELS


def find_subject_file(subj):
    """Encuentra el archivo crudo de un sujeto (match exacto del token antes de '_')."""
    cands = []
    for f in RAW_DIR.iterdir():
        if f.suffix.lower() not in (".bdf", ".set"):
            continue
        if f.stem.split("_")[0] == subj:
            cands.append(f)
    if not cands:
        raise FileNotFoundError(f"No se encontro archivo crudo para {subj}")
    # preferir .bdf si hay ambos
    cands.sort(key=lambda p: (p.suffix.lower() != ".bdf"))
    return cands[0]


def _rama_from_name(fname):
    for tok in fname.split("_"):
        if tok in ("R1", "R2"):
            return tok[1]
    return None


def _load_bdf(path):
    raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    sf = raw.info["sfreq"]
    picks = [raw.ch_names.index(l) for l in EMG_LABELS]
    data = raw.get_data(picks=picks)  # V, shape (4, n)
    ev = mne.find_events(raw, stim_channel="Status", verbose="ERROR",
                         shortest_event=1, consecutive=True)
    ev = ev[ev[:, 2] < 256]
    events = np.column_stack([ev[:, 0], ev[:, 2]]).astype(int)
    return data, sf, events


def _h5_deref_scalar(f, ref):
    v = np.array(f[ref]).squeeze()
    return float(v)


def _h5_str(f, ref):
    arr = np.array(f[ref]).flatten()
    return "".join(chr(int(c)) for c in arr)


def _load_v73_set(path):
    """Lee un .set EEGLAB guardado como MAT v7.3 (HDF5)."""
    with h5py.File(path, "r") as f:
        eeg = f["EEG"] if "EEG" in f else f  # algunos .set guardan los campos en la raiz
        sf = float(np.array(eeg["srate"]).flatten()[0])
        data = np.array(eeg["data"])           # (n_samp, n_ch)
        if data.shape[0] < data.shape[1]:
            data = data.T                       # asegurar (n_samp, n_ch)
        # etiquetas de canal (object array, puede venir (1,N))
        labels = [_h5_str(f, ref) for ref in np.array(eeg["chanlocs"]["labels"]).flatten()]
        idx = [labels.index(l) for l in EMG_LABELS]
        emg = data[:, idx].T.astype(float)      # (4, n)  en microV (EEGLAB) -> a V
        emg = emg * 1e-6
        # eventos
        types = eeg["event"]["type"]
        lats = eeg["event"]["latency"]
        codes, samples = [], []
        for i in range(types.shape[0]):
            tref, lref = types[i][0], lats[i][0]
            tobj = np.array(f[tref]).squeeze()
            # EEGLAB v7.3: type numerico -> float64; type char -> uint16 (ASCII)
            if tobj.dtype.kind == "f":
                code = int(round(float(tobj)))
            elif tobj.dtype.kind in "iu":
                s = "".join(chr(int(c)) for c in np.atleast_1d(tobj).flatten())
                if not s.strip().lstrip("-").isdigit():
                    continue
                code = int(s)
            else:
                continue
            lat = float(np.array(f[lref]).squeeze())
            codes.append(code)
            samples.append(int(round(lat)) - 1)  # 1-idx -> 0-idx
        events = np.column_stack([samples, codes]).astype(int)
        events = events[(events[:, 1] >= 0) & (events[:, 1] < 256)]
    return emg, sf, events


def load_emg(subj):
    """Carga EMG de un sujeto. Devuelve dict con data(4,n), sfreq, events, rama, file."""
    path = find_subject_file(subj)
    rama = _rama_from_name(path.stem)
    if path.suffix.lower() == ".bdf":
        data, sf, events = _load_bdf(path)
    else:
        try:
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
            picks = [raw.ch_names.index(l) for l in EMG_LABELS]
            data = raw.get_data(picks=picks)
            ev, eid = mne.events_from_annotations(raw, verbose="ERROR")
            inv = {v: k for k, v in eid.items()}
            codes = []
            for code in ev[:, 2]:
                lab = inv[code]
                codes.append(int(lab) if str(lab).lstrip("-").isdigit() else -1)
            events = np.column_stack([ev[:, 0], codes]).astype(int)
            events = events[events[:, 1] >= 0]
            sf = raw.info["sfreq"]
        except NotImplementedError:
            data, sf, events = _load_v73_set(path)
    return {"subj": subj, "data": data, "sfreq": sf,
            "events": events, "rama": rama, "file": path.name}


if __name__ == "__main__":
    # Prueba rapida con un bdf, el .set v7.3 y conteo de eventos
    for s in ["M3", "PS1", "M2"]:
        d = load_emg(s)
        codes = d["events"][:, 1]
        from collections import Counter
        cc = {k: v for k, v in sorted(Counter(codes).items()) if k <= 6}
        print(f"{s:4} file={d['file']:<32} rama=R{d['rama']} sf={d['sfreq']:.0f} "
              f"emg={d['data'].shape} codes(<=6)={cc}")
