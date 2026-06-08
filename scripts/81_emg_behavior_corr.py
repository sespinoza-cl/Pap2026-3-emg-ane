"""Does the bout-1 EMG compensation relate to behavior?

Behavioral anesthesia effect in the CHEW conditions (ane2=Anesthesia+chew,
ane4=Placebo+chew; condition codes are consistent across subjects) vs the bout-1
EMG compensation (median cycle amplitude ANE-PLA). Spearman correlations.

Compensatory hypothesis: if the motor compensation preserves performance, the
behavioral anesthesia effect should be ~0 (replicating the parent paper) and need
not scale with compensation; we test this explicitly and report it honestly.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from config import OUT

BEH = r"D:\Exp1\Exp1\Behavior\All_36\Exp1_beh_all.csv"
ANE_CODE, PLA_CODE = 2.0, 4.0  # ane2 = Anesthesia+chew, ane4 = Placebo+chew


def beh_measures():
    df = pd.read_csv(BEH)
    df["subj"] = df["participant"].str.split("_").str[0]
    rows = []
    for subj, g in df.groupby("subj"):
        rec = {"subj": subj}
        for code, tag in [(ANE_CODE, "ANE"), (PLA_CODE, "PLA")]:
            c = g[g["ane"] == code]
            # 2-back
            tb = c[c["k_tb.rt"].notna()]
            rt = tb["k_tb.rt"].copy()
            if len(rt) > 10:
                lo, hi = rt.quantile([0.01, 0.99])
                rtc = tb[(tb["ans_tb"] == 1) & (tb["k_tb.rt"].between(lo, hi))]["k_tb.rt"]
                rec[f"tb_rt_{tag}"] = rtc.mean() * 1000  # ms
                rec[f"tb_acc_{tag}"] = tb["ans_tb"].mean() * 100
            # VO
            vo = c[c["k_vo.rt"].notna()]
            if len(vo) > 10:
                lo, hi = vo["k_vo.rt"].quantile([0.01, 0.99])
                rtc = vo[(vo["ans_vo"] == 1) & (vo["k_vo.rt"].between(lo, hi))]["k_vo.rt"]
                rec[f"vo_rt_{tag}"] = rtc.mean() * 1000
                rec[f"vo_acc_{tag}"] = vo["ans_vo"].mean() * 100
        rows.append(rec)
    b = pd.DataFrame(rows)
    for m in ["tb_rt", "tb_acc", "vo_rt", "vo_acc"]:
        b[f"d_{m}"] = b[f"{m}_ANE"] - b[f"{m}_PLA"]
    return b


def main():
    rob = pd.read_csv(OUT / "robustness_table.csv")
    rob = rob[rob["auto_ok"]].copy()
    rob["emg_comp"] = rob["ANE_b1"] - rob["PLA_b1"]   # uV, bout-1 compensation
    beh = beh_measures()
    m = rob[["subj", "emg_comp"]].merge(beh, on="subj")
    print(f"N con EMG+conducta: {len(m)}")

    # efecto conductual de anestesia (debe ser ~0 segun Paper1)
    print("\n=== Behavioral anesthesia effect (ANE-PLA, chew) ===")
    for d, lab, unit in [("d_tb_rt", "2-back RT", "ms"), ("d_tb_acc", "2-back acc", "%"),
                         ("d_vo_rt", "VO RT", "ms"), ("d_vo_acc", "VO acc", "%")]:
        x = m[d].dropna()
        t = stats.wilcoxon(x).pvalue if len(x) > 5 else np.nan
        print(f"  {lab:12}: mean={x.mean():+.2f} {unit}  (Wilcoxon p={t:.3f})")

    # correlaciones EMG-compensacion vs conducta
    print("\n=== Spearman: bout-1 EMG compensation vs behavior ===")
    targets = [("d_tb_rt", "Δ 2-back RT"), ("d_tb_acc", "Δ 2-back acc"),
               ("d_vo_rt", "Δ VO RT"), ("d_vo_acc", "Δ VO acc"),
               ("tb_acc_ANE", "2-back acc (ANE)"), ("tb_rt_ANE", "2-back RT (ANE)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (col, lab) in zip(axes.ravel(), targets):
        d = m[["emg_comp", col]].dropna()
        rho, p = stats.spearmanr(d["emg_comp"], d[col])
        ax.scatter(d["emg_comp"], d[col], s=20, color="#34495e")
        ax.set_xlabel("Bout-1 EMG compensation ANE−PLA (µV)")
        ax.set_ylabel(lab)
        ax.set_title(f"{lab}\nSpearman ρ={rho:.2f}, p={p:.3f} (n={len(d)})", fontsize=8)
        print(f"  {lab:18}: rho={rho:+.2f}, p={p:.3f}, n={len(d)}")
    fig.suptitle("Bout-1 EMG compensation vs behavioral measures (exploratory)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_emg_behavior.png"); plt.close(fig)
    m.to_csv(OUT / "emg_behavior_table.csv", index=False)
    print("\nSaved:", OUT / "fig_emg_behavior.png", "and emg_behavior_table.csv")


if __name__ == "__main__":
    main()
