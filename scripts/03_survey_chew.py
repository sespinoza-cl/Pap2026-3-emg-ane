"""Censo de CHEW/Raw_full: canales, eventos, duracion por archivo."""
import mne
from pathlib import Path
from collections import Counter

CHEW = Path(r"D:\Exp1\Exp1\EEG\CHEW\Raw_full")
files = sorted(CHEW.glob("*.set"))
print(f"Total .set en CHEW/Raw_full: {len(files)}\n")
print(f"{'archivo':<24} {'nch':>4} {'sfreq':>6} {'dur_s':>7}  eventos(no-boundary)")
print("-"*90)
for f in files:
    raw = mne.io.read_raw_eeglab(f, preload=False, verbose="ERROR")
    desc = Counter(d for d in raw.annotations.description if d != "boundary")
    nb = sum(1 for d in raw.annotations.description if d == "boundary")
    extra = raw.ch_names[64:] if len(raw.ch_names) > 64 else ""
    print(f"{f.stem:<24} {len(raw.ch_names):>4} {raw.info['sfreq']:>6.0f} "
          f"{raw.n_times/raw.info['sfreq']:>7.1f}  {dict(sorted(desc.items()))} bnd={nb} {extra}")
