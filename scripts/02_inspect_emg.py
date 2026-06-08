"""Inspecciona los archivos EMG previos y los .mat de SNR/listas."""
import mne, numpy as np, scipy.io as sio
from pathlib import Path
from collections import Counter

def show_set(path, label):
    print(f"\n===== {label}: {Path(path).name} =====")
    raw = mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
    print("sfreq:", raw.info["sfreq"], " nch:", len(raw.ch_names),
          " dur(s):", round(raw.n_times/raw.info['sfreq'],1))
    print("ch_names:", raw.ch_names)
    desc = Counter(raw.annotations.description)
    print("events:", dict(sorted(desc.items())))

# EMG Step1 (continuo) y Step2 (limpio)
show_set(r"D:\Exp1\Exp1\EMG\Step1\M1_A1_cont.set", "Step1 A1 cont")
show_set(r"D:\Exp1\Exp1\EMG\Step1\M1_A2_cont.set", "Step1 A2 cont")
show_set(r"D:\Exp1\Exp1\EMG\Step1\M1_Rs_set.set", "Step1 Rs")
show_set(r"D:\Exp1\Exp1\EMG\Step2\M1_A1_clean.set", "Step2 A1 clean")
# CHEW raw full
show_set(r"D:\Exp1\Exp1\EEG\CHEW\Raw_full\M1_R2_ch_step0.set", "CHEW raw full")

print("\n\n===== .MAT FILES =====")
for mf in ["EMG_SNR.mat", "lista30.mat", "lista32.mat", "lista36.mat"]:
    p = Path(r"D:\Exp1\Exp1\EMG") / mf
    try:
        m = sio.loadmat(p)
        print(f"\n--- {mf} ---")
        for k, v in m.items():
            if k.startswith("__"): continue
            arr = np.array(v)
            print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")
            if arr.size <= 80:
                print("    ", arr.squeeze())
    except Exception as e:
        print(f"  ERROR {mf}: {e}")
