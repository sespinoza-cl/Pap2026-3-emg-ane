"""Confounds de orden / arrastre (crossover) sobre el efecto del bout 1.

R1: Anestesia 1deg, Placebo 2deg (placebo podria tener arrastre residual).
R2: Placebo 1deg (limpio), Anestesia 2deg.

Tests:
  1. ANOVA mixto bout1 med_amp: within=condicion, between=rama -> la interaccion
     condicion*rama es la firma de arrastre/efecto de orden sobre el tratamiento.
  2. Entre-sujetos: PLA bout1 R1(post-anestesia) vs R2(limpio) = ARRASTRE.
                    ANE bout1 R1(1deg) vs R2(2deg)            = ORDEN sobre anestesia.
  3. Intra-rama: ANE vs PLA bout1 pareado, por separado en R1 y R2 (¿efecto en ambos?).
  4. Efecto de PERIODO: 1er vs 2do bloque (amplitud), intra-sujeto (habituacion/fatiga).
"""
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from config import OUT

FEAT = "med_amp"
SCALE = 1e6  # uV


def main():
    qc = pd.read_csv(OUT / "qc_table.csv")
    inc = qc[qc["auto_ok"]][["subj", "rama"]]
    df = pd.read_csv(OUT / "bout_features.csv").merge(inc, on="subj")
    b1 = df[df.bout == 1].copy()
    b1[FEAT] = b1[FEAT] * SCALE
    w = b1.pivot_table(index=["subj", "rama"], columns="cond", values=FEAT).reset_index()
    print("N por rama:", w.rama.value_counts().to_dict())

    # 1. ANOVA mixto (within=cond, between=rama)
    long = b1[["subj", "rama", "cond", FEAT]].rename(columns={FEAT: "y"})
    aov = pg.mixed_anova(data=long, dv="y", within="cond", between="rama", subject="subj")
    print("\n=== ANOVA mixto bout1 (within=cond, between=rama) ===")
    pcol = "p_unc" if "p_unc" in aov.columns else "p-unc"
    keep = [c for c in ["Source", "DF1", "DF2", "F", pcol, "np2"] if c in aov.columns]
    print(aov[keep].round(4).to_string(index=False))

    # 2. entre-sujetos
    pla_r1 = w[w.rama == "R1"]["PLA"].values   # placebo 2deg (post-anestesia)
    pla_r2 = w[w.rama == "R2"]["PLA"].values   # placebo 1deg (limpio)
    ane_r1 = w[w.rama == "R1"]["ANE"].values   # anestesia 1deg
    ane_r2 = w[w.rama == "R2"]["ANE"].values   # anestesia 2deg
    u_pla = stats.mannwhitneyu(pla_r1, pla_r2)
    u_ane = stats.mannwhitneyu(ane_r1, ane_r2)
    print("\n=== Entre-sujetos (bout1, uV) ===")
    print(f"ARRASTRE  PLA: R1(2deg,post-ane) med={np.median(pla_r1):.1f} vs "
          f"R2(1deg,limpio) med={np.median(pla_r2):.1f}  Mann-Whitney p={u_pla.pvalue:.3f}")
    print(f"ORDEN     ANE: R1(1deg) med={np.median(ane_r1):.1f} vs "
          f"R2(2deg) med={np.median(ane_r2):.1f}  Mann-Whitney p={u_ane.pvalue:.3f}")

    # 3. intra-rama: ANE vs PLA bout1
    print("\n=== Efecto ANE-PLA bout1 por rama (pareado) ===")
    for r in ["R1", "R2"]:
        sub = w[w.rama == r]
        diff = sub["ANE"].values - sub["PLA"].values
        wp = stats.wilcoxon(sub["ANE"], sub["PLA"]).pvalue
        dz = diff.mean() / diff.std(ddof=1)
        print(f"{r} (n={len(sub)}): dif_media={diff.mean():.1f} uV, dz={dz:.2f}, "
              f"Wilcoxon p={wp:.3f}, {100*np.mean(diff>0):.0f}% ANE>PLA")

    # 4. efecto de PERIODO (1er vs 2do bloque), intra-sujeto
    #    R1: periodo1=ANE, periodo2=PLA ; R2: periodo1=PLA, periodo2=ANE
    per1, per2 = [], []
    for _, row in w.iterrows():
        if row["rama"] == "R1":
            per1.append(row["ANE"]); per2.append(row["PLA"])
        else:
            per1.append(row["PLA"]); per2.append(row["ANE"])
    per1, per2 = np.array(per1), np.array(per2)
    wp = stats.wilcoxon(per1, per2).pvalue
    print("\n=== Efecto de PERIODO (bout1, intra-sujeto) ===")
    print(f"Periodo1 med={np.median(per1):.1f} vs Periodo2 med={np.median(per2):.1f} uV, "
          f"Wilcoxon p={wp:.3f}  (habituacion/fatiga inespecifica de condicion)")

    # ---- figura: cond x rama ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from fig_style import RC, FIGSIZE, ANEC, PLAC, ANED, PLAD, SUPPL, save_fig
    plt.rcParams.update(RC)
    b1["Condition"] = b1["cond"].map({"ANE": "Anesthesia", "PLA": "Placebo"})
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(data=b1, x="rama", y=FEAT, hue="Condition",
                palette=[ANEC, PLAC], showfliers=False, ax=ax)
    sns.stripplot(data=b1, x="rama", y=FEAT, hue="Condition",
                  palette=[ANED, PLAD], dodge=True, size=3, alpha=0.6,
                  ax=ax, legend=False)
    ax.set_xticklabels(["R1 (Anesth.→Plac.)", "R2 (Plac.→Anesth.)"])
    ax.set_ylabel("Median cycle amplitude, bout 1 (µV)")
    ax.set_xlabel("Sequence")
    ax.legend(title="")
    fig.tight_layout()
    save_fig(fig, "s7_carryover", SUPPL)
    print("\nSaved: s7_carryover")


if __name__ == "__main__":
    main()
