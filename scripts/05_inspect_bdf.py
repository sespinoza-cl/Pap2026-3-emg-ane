"""Inspecciona los BDF crudos de Raw/All_36: canales, sfreq, eventos (status)."""
import mne, numpy as np
from pathlib import Path
from collections import Counter

RAW = Path(r"D:\Exp1\Exp1\Raw data\Raw\All_36")
files = sorted(RAW.glob("*"))
print("Archivos en All_36:")
for f in files:
    print("  ", f.name)

# Leer un BDF representativo
bdf = RAW / "M1_R2_231023.bdf"
print(f"\n===== Leyendo {bdf.name} =====")
raw = mne.io.read_raw_bdf(bdf, preload=False, verbose="ERROR")
print("sfreq:", raw.info["sfreq"], " nch:", len(raw.ch_names),
      " dur(s):", round(raw.n_times/raw.info["sfreq"],1))
print("\nch_names:")
for i, ch in enumerate(raw.ch_names):
    print(f"  {i:>3} (1-idx {i+1:>3}): {ch}  [{raw.get_channel_types()[i]}]")

# Eventos desde status channel
print("\n=== EVENTOS (find_events sobre Status) ===")
try:
    ev = mne.find_events(raw, stim_channel="Status", verbose="ERROR")
    print("n_events:", len(ev))
    print("codigos y conteo:", dict(sorted(Counter(ev[:,2]).items())))
    print("primeros 20 (sample, _, code):")
    for r in ev[:20]:
        print("   t=%.2fs  code=%d" % (r[0]/raw.info["sfreq"], r[2]))
except Exception as e:
    print("find_events fallo:", e)
    print("annotations:", Counter(raw.annotations.description))
