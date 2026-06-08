"""Features EMG por BOUT (1-4) dentro de cada condicion, lado elegido.

Los 4 bouts estan en orden cronologico desde la aplicacion del spray:
bout 1 = inmediatamente despues (anestesia maxima), bout 4 = ~13 min despues
(posible washout). Permite testear si el efecto de la anestesia es mayor temprano.
"""
import numpy as np
import pandas as pd
from config import DERIV, OUT
from emg_features import all_features
from emg_tf import reshape_bouts


def main():
    qc = pd.read_csv(OUT / "qc_table.csv")
    inc = qc[qc["auto_ok"]]
    rows = []
    for _, q in inc.iterrows():
        subj, side = q["subj"], q["side"]
        d = np.load(DERIV / f"{subj}_emg.npz")
        fs = float(d["fs"])
        rs = d[f"RS_{side}"]
        for cond in ("ANE", "PLA"):
            sig = d[f"{cond}_{side}"]
            bouts = reshape_bouts(sig, fs, n_bouts=4)
            for bi, b in enumerate(bouts, start=1):
                feats = all_features(b, fs, rs_sig=rs)
                rows.append(dict(subj=subj, side=side, cond=cond, bout=bi, **feats))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bout_features.csv", index=False)
    print("Guardado:", OUT / "bout_features.csv", " filas:", len(df),
          " sujetos:", df.subj.nunique())
    # resumen medio por bout/cond para metricas clave
    for f in ["rms_amp", "totpow", "med_amp", "MDF", "chew_rate"]:
        t = df.pivot_table(index="bout", columns="cond", values=f, aggfunc="mean")
        t["diff_ANE-PLA"] = t["ANE"] - t["PLA"]
        print(f"\n== {f} (media por bout) ==")
        print(t.to_string())


if __name__ == "__main__":
    main()
