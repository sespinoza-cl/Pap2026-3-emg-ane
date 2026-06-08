"""Control de calidad y seleccion de lado (bipolar) por sujeto.

Seleccion de lado (criterio fisiologico, robusto a electrodos flotantes):
se exige que el lado tenga EMG plausible -> MDF en rango y tasa en rango en
AMBAS condiciones (un electrodo malo da MDF baja por deriva de baja frecuencia y
puede inflar el SNR si su reposo esta casi mudo). Entre los lados que pasan la
puerta fisiologica se elige el de mayor SNR minimo entre condiciones. Si ningun
lado pasa, se marca el sujeto.

QC automatico (flags): SNR bajo, tasa fuera de rango, MDF no fisiologica,
asimetria de SNR entre condiciones (posible fallo de electrodo en un bloque).
La inclusion final la confirma el usuario con las figuras/diagnosticos de QC.
"""
import numpy as np
import pandas as pd
from config import OUT, LISTA30

SNR_MIN = 12.0          # dB
ASYM_MAX = 8.0          # dB
RATE_RANGE = (0.6, 2.6)  # Hz
MDF_RANGE = (55.0, 170.0)  # Hz: EMG de masetero real (descarta deriva LF)


def main():
    df = pd.read_csv(OUT / "features_all.csv")
    pv = df.pivot_table(index=["subj", "rama", "side"], columns="cond",
                        values=["snr_db", "chew_rate", "MDF"]).reset_index()
    pv.columns = ["_".join([c for c in col if c]).strip("_") for col in pv.columns]

    def physio_ok(r):
        return (MDF_RANGE[0] <= r["MDF_ANE"] <= MDF_RANGE[1] and
                MDF_RANGE[0] <= r["MDF_PLA"] <= MDF_RANGE[1] and
                RATE_RANGE[0] <= r["chew_rate_ANE"] <= RATE_RANGE[1] and
                RATE_RANGE[0] <= r["chew_rate_PLA"] <= RATE_RANGE[1])

    rows = []
    for subj, g in pv.groupby("subj"):
        cand = []
        for _, r in g.iterrows():
            snr_min = np.nanmin([r["snr_db_ANE"], r["snr_db_PLA"]])
            cand.append((physio_ok(r), snr_min, r))
        # preferir lados con EMG fisiologico; dentro, mayor SNR minimo
        cand.sort(key=lambda c: (c[0], c[1]), reverse=True)
        gate_ok, snr_min, r = cand[0]
        best = dict(side=r["side"], rama=r["rama"],
                    snr_ane=r["snr_db_ANE"], snr_pla=r["snr_db_PLA"], snr_min=snr_min,
                    rate_ane=r["chew_rate_ANE"], rate_pla=r["chew_rate_PLA"],
                    mdf_ane=r["MDF_ANE"], mdf_pla=r["MDF_PLA"], gate=gate_ok)
        flags = []
        if not best["gate"]:
            flags.append("sin_lado_fisiologico")
        if best["snr_min"] < SNR_MIN:
            flags.append(f"SNR<{SNR_MIN}")
        if abs(best["snr_ane"] - best["snr_pla"]) > ASYM_MAX:
            flags.append("asimetria_SNR")
        rows.append(dict(subj=subj, rama=best["rama"], side=best["side"],
                         snr_ANE=round(best["snr_ane"], 1), snr_PLA=round(best["snr_pla"], 1),
                         snr_min=round(best["snr_min"], 1),
                         rate_ANE=round(best["rate_ane"], 2), rate_PLA=round(best["rate_pla"], 2),
                         MDF_ANE=round(best["mdf_ane"], 0), MDF_PLA=round(best["mdf_pla"], 0),
                         auto_ok=(len(flags) == 0), flags=";".join(flags) if flags else "",
                         en_lista30=subj in LISTA30))
    qc = pd.DataFrame(rows).sort_values("snr_min")
    qc.to_csv(OUT / "qc_table.csv", index=False)
    print(qc.to_string(index=False))
    print(f"\nAuto-OK: {qc['auto_ok'].sum()}/{len(qc)} sujetos")
    print("Marcados:", qc.loc[~qc["auto_ok"], "subj"].tolist())
    print("Lado elegido: L=%d  R=%d" % ((qc.side=="L").sum(), (qc.side=="R").sum()))


if __name__ == "__main__":
    main()
