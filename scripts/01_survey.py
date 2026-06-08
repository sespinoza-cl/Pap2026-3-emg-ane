"""Censo de los 36 archivos: n_canales, nombres extra (>64), sfreq, duracion."""
import mne
from pathlib import Path
from collections import Counter

RAW = Path(r"D:\Exp1\Exp1\EEG\DATA\Raw_Full")
files = sorted(RAW.glob("*.set"))
print(f"Total archivos: {len(files)}\n")
print(f"{'archivo':<22} {'nch':>4} {'sfreq':>6} {'dur_s':>8}   canales_extra(>64)")
print("-"*100)
nch_counter = Counter()
for f in files:
    raw = mne.io.read_raw_eeglab(f, preload=False, verbose="ERROR")
    nch = len(raw.ch_names)
    nch_counter[nch] += 1
    extra = raw.ch_names[64:] if nch > 64 else []
    print(f"{f.stem:<22} {nch:>4} {raw.info['sfreq']:>6.0f} {raw.n_times/raw.info['sfreq']:>8.1f}   {extra}")

print("\nDistribucion de n_canales:", dict(nch_counter))
