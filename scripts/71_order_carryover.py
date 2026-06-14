"""Order and carryover confounds (crossover design) on the bout-1 effect.

R1: Anesthesia first, Placebo second (placebo may retain residual carryover).
R2: Placebo first (clean baseline), Anesthesia second.

Tests:
  1. Mixed ANOVA on bout-1 med_amp: within=condition, between=arm -> condition*arm
     interaction is the signature of carryover / order effect on treatment.
  2. Between-subject: PLA bout1 R1 (post-anesthesia) vs R2 (clean) = CARRYOVER.
                      ANE bout1 R1 (first) vs R2 (second)          = ORDER effect.
  3. Within-arm: ANE vs PLA bout1 paired, separately for R1 and R2 (effect in both?).
  4. PERIOD effect: 1st vs 2nd block (amplitude), within-subject (habituation/fatigue).
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

    # 1. Mixed ANOVA (within=condition, between=arm)
    long = b1[["subj", "rama", "cond", FEAT]].rename(columns={FEAT: "y"})
    aov = pg.mixed_anova(data=long, dv="y", within="cond", between="rama", subject="subj")
    print("\n=== Mixed ANOVA bout1 (within=cond, between=arm) ===")
    pcol = "p_unc" if "p_unc" in aov.columns else "p-unc"
    keep = [c for c in ["Source", "DF1", "DF2", "F", pcol, "np2"] if c in aov.columns]
    print(aov[keep].round(4).to_string(index=False))

    # 2. between-subjects
    pla_r1 = w[w.rama == "R1"]["PLA"].values   # placebo 2nd (post-anesthesia)
    pla_r2 = w[w.rama == "R2"]["PLA"].values   # placebo 1st (clean baseline)
    ane_r1 = w[w.rama == "R1"]["ANE"].values   # anesthesia 1st
    ane_r2 = w[w.rama == "R2"]["ANE"].values   # anesthesia 2nd
    u_pla = stats.mannwhitneyu(pla_r1, pla_r2)
    u_ane = stats.mannwhitneyu(ane_r1, ane_r2)
    print("\n=== Between-subjects (bout1, uV) ===")
    print(f"CARRYOVER PLA: R1(2nd,post-ane) med={np.median(pla_r1):.1f} vs "
          f"R2(1st,clean) med={np.median(pla_r2):.1f}  Mann-Whitney p={u_pla.pvalue:.3f}")
    print(f"ORDER     ANE: R1(1st) med={np.median(ane_r1):.1f} vs "
          f"R2(2nd) med={np.median(ane_r2):.1f}  Mann-Whitney p={u_ane.pvalue:.3f}")

    # 3. within-arm: ANE vs PLA bout 1
    print("\n=== ANE-PLA effect bout1 by arm (paired) ===")
    for r in ["R1", "R2"]:
        sub = w[w.rama == r]
        diff = sub["ANE"].values - sub["PLA"].values
        wp = stats.wilcoxon(sub["ANE"], sub["PLA"]).pvalue
        dz = diff.mean() / diff.std(ddof=1)
        print(f"{r} (n={len(sub)}): mean_diff={diff.mean():.1f} uV, dz={dz:.2f}, "
              f"Wilcoxon p={wp:.3f}, {100*np.mean(diff>0):.0f}% ANE>PLA")

    # 4. period effect (1st vs 2nd block), within-subject
    #    R1: period1=ANE, period2=PLA ; R2: period1=PLA, period2=ANE
    per1, per2 = [], []
    for _, row in w.iterrows():
        if row["rama"] == "R1":
            per1.append(row["ANE"]); per2.append(row["PLA"])
        else:
            per1.append(row["PLA"]); per2.append(row["ANE"])
    per1, per2 = np.array(per1), np.array(per2)
    wp = stats.wilcoxon(per1, per2).pvalue
    print("\n=== Period effect (bout1, within-subject) ===")
    print(f"Period1 med={np.median(per1):.1f} vs Period2 med={np.median(per2):.1f} uV, "
          f"Wilcoxon p={wp:.3f}  (non-specific habituation/fatigue unrelated to condition)")

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
