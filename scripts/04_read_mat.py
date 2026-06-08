"""Lee los .mat v7.3 (HDF5): EMG_SNR, listas, AllSubjects_Results."""
import h5py, numpy as np
from pathlib import Path

EMG = Path(r"D:\Exp1\Exp1\EMG")

def h5str(f, ref):
    """Decodifica una cadena MATLAB (array de uint16) desde una referencia."""
    obj = f[ref]
    return "".join(chr(c) for c in np.array(obj).flatten())

def dump_tree(name, obj):
    print("  ", name, getattr(obj, "shape", ""), getattr(obj, "dtype", ""))

for mf in ["lista36.mat", "lista30.mat", "lista32.mat"]:
    print(f"\n===== {mf} =====")
    with h5py.File(EMG/mf, "r") as f:
        for key in f.keys():
            if key == "#refs#": continue
            ds = f[key]
            print(f" var '{key}': shape={ds.shape}")
            # celda de cadenas
            try:
                vals = []
                for ref in np.array(ds).flatten():
                    vals.append(h5str(f, ref))
                print("   ->", vals)
            except Exception as e:
                print("   (no cell strings):", e)

print("\n\n===== EMG_SNR.mat =====")
with h5py.File(EMG/"EMG_SNR.mat", "r") as f:
    def walk(g, pre=""):
        for k in g.keys():
            if k == "#refs#": continue
            item = g[k]
            if isinstance(item, h5py.Group):
                print(f"{pre}{k}/ (group)")
                walk(item, pre+"  ")
            else:
                print(f"{pre}{k}: shape={item.shape} dtype={item.dtype}")
    walk(f)
