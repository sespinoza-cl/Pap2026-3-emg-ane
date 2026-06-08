"""Verifica duracion de bouts (code1->code2), reposo (3-6), y los .set especiales."""
import mne, numpy as np
from pathlib import Path
from collections import Counter

RAW = Path(r"D:\Exp1\Exp1\Raw data\Raw\All_36")

def get_events(raw):
    try:
        return mne.find_events(raw, stim_channel="Status", verbose="ERROR", shortest_event=1)
    except Exception:
        # eeglab: usar annotations
        ev, _ = mne.events_from_annotations(raw, verbose="ERROR")
        return ev

# --- BDF: timing de bouts ---
bdf = RAW / "M3_R1_201123_.bdf"
raw = mne.io.read_raw_bdf(bdf, preload=False, verbose="ERROR")
sf = raw.info["sfreq"]
ev = mne.find_events(raw, stim_channel="Status", verbose="ERROR")
# limpiar high-byte (>255)
ev = ev[ev[:,2] < 256]
c1 = ev[ev[:,2]==1][:,0]/sf
c2 = ev[ev[:,2]==2][:,0]/sf
rs = ev[np.isin(ev[:,2],[3,4,5,6])][:,0]/sf
print(f"=== {bdf.name} (R1) sf={sf} ===")
print("code1 (chew onset) times:", np.round(c1,1))
print("code2 (chew offset) times:", np.round(c2,1))
print("dur code1->siguiente code2:", np.round(c2-c1,1) if len(c1)==len(c2) else "len mismatch")
print("RS markers (3-6) times:", np.round(rs,1))

# --- .set especiales ---
for setf in ["PS1_R1_240523_merged.set", "M2_R2_261023.set"]:
    p = RAW / setf
    print(f"\n=== {setf} ===")
    r = mne.io.read_raw_eeglab(p, preload=False, verbose="ERROR")
    print("sf:", r.info["sfreq"], " nch:", len(r.ch_names), " dur:", round(r.n_times/r.info['sfreq'],1))
    print("last 10 ch:", r.ch_names[-10:])
    desc = Counter(r.annotations.description)
    print("eventos:", dict(sorted(desc.items(), key=lambda kv: str(kv[0]))))
