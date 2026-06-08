"""Inspecciona un archivo .set para entender canales, eventos y estructura."""
import mne
import numpy as np
from pathlib import Path

RAW = Path(r"D:\Exp1\Exp1\EEG\DATA\Raw_Full")
f = RAW / "PS1_R1_raw_step0.set"

raw = mne.io.read_raw_eeglab(f, preload=False, verbose="ERROR")
print("=== INFO ===")
print("sfreq:", raw.info["sfreq"])
print("n_times:", raw.n_times, "  dur(s):", raw.n_times / raw.info["sfreq"])
print("n_channels:", len(raw.ch_names))
print("\n=== CH NAMES (todos) ===")
for i, ch in enumerate(raw.ch_names):
    print(f"{i:>3} (1-idx {i+1:>3}): {ch}")

print("\n=== EVENTS / ANNOTATIONS ===")
ann = raw.annotations
print("n_annotations:", len(ann))
# Distribucion de descripciones
from collections import Counter
desc = Counter(ann.description)
print("Tipos de evento y conteo:")
for k, v in sorted(desc.items()):
    print(f"  '{k}': {v}")

print("\n=== Primeros 40 eventos (onset, dur, desc) ===")
for i in range(min(40, len(ann))):
    print(f"  t={ann.onset[i]:8.2f}s  dur={ann.duration[i]:6.2f}  '{ann.description[i]}'")
