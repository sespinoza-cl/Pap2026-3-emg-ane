import h5py, numpy as np
from pathlib import Path
p = Path(r"D:\Exp1\Exp1\Raw data\Raw\All_36\PS1_R1_240523_merged.set")
with h5py.File(p, "r") as f:
    def walk(g, pre="", depth=0):
        if depth > 2: return
        for k in g.keys():
            if k == "#refs#": continue
            item = g[k]
            if isinstance(item, h5py.Group):
                print(f"{pre}{k}/ (group) keys={list(item.keys())[:15]}")
                walk(item, pre+"  ", depth+1)
            else:
                print(f"{pre}{k}: shape={item.shape} dtype={item.dtype}")
    walk(f)
    eeg = f["EEG"]
    print("\nsrate:", np.array(eeg["srate"]).flatten())
    print("nbchan:", np.array(eeg["nbchan"]).flatten())
    print("pnts:", np.array(eeg["pnts"]).flatten())
    print("data field:", eeg["data"], eeg["data"].shape if hasattr(eeg["data"],'shape') else "")
    # event struct
    if "event" in eeg:
        evt = eeg["event"]
        print("event keys:", list(evt.keys()))
        print("event type shape:", evt["type"].shape, " latency shape:", evt["latency"].shape)
