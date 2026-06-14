"""Bout-resolved statistics: is the anesthesia effect larger early and does it decay?

Three tests per metric (on the paired difference d = ANE - PLA per bout):
  1. Paired Wilcoxon per bout (ANE vs PLA) -> is there an effect at bout 1?
  2. Decay trend: per-subject slope of d vs bout -> does the effect decay? (test vs 0)
  3. Repeated-measures ANOVA 2(condition) x 4(bout) -> condition*bout interaction.
FDR (Benjamini-Hochberg) applied to bout-1 p-values across metrics.
"""
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from config import OUT

FEATURES = ["rms_amp", "totpow", "med_amp", "med_area", "p_20_60", "p_60_150",
            "MDF", "MNF", "chew_rate", "cv_ipi", "duty"]


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for rank in range(n-1, -1, -1):
        i = order[rank]; prev = min(prev, p[i]*n/(rank+1)); adj[i] = prev
    return adj


def main():
    df = pd.read_csv(OUT / "bout_features.csv")
    print("Subjects:", df.subj.nunique(), " bouts/cond:", sorted(df.bout.unique()))
    rows, inter_rows = [], []
    for feat in FEATURES:
        # subject x bout matrix for each condition
        a = df[df.cond == "ANE"].pivot_table(index="subj", columns="bout", values=feat)
        p = df[df.cond == "PLA"].pivot_table(index="subj", columns="bout", values=feat)
        common = a.dropna().index.intersection(p.dropna().index)
        a, p = a.loc[common], p.loc[common]
        d = a - p  # paired difference per bout
        n = len(common)

        # 1. paired per bout
        bout_p, bout_dz = {}, {}
        for b in [1, 2, 3, 4]:
            x, y = a[b].values, p[b].values
            try:
                wp = stats.wilcoxon(x, y).pvalue
            except ValueError:
                wp = np.nan
            dz = (x-y).mean()/(x-y).std(ddof=1) if (x-y).std(ddof=1) > 0 else np.nan
            bout_p[b] = wp; bout_dz[b] = dz

        # 2. decay trend: per-subject slope of d vs bout
        slopes = np.array([np.polyfit([1, 2, 3, 4], d.loc[s].values, 1)[0] for s in common])
        tr = stats.wilcoxon(slopes).pvalue if np.any(slopes != 0) else np.nan
        slope_mean = slopes.mean()

        # 3. RM-ANOVA 2x4 (interaction)
        long = df[df.subj.isin(common)][["subj", "cond", "bout", feat]].rename(columns={feat: "y"})
        try:
            aov = pg.rm_anova(data=long, dv="y", within=["cond", "bout"], subject="subj",
                              detailed=True)
            pcol = "p_unc" if "p_unc" in aov.columns else "p-unc"
            gg = "p_GG_corr" if "p_GG_corr" in aov.columns else pcol
            row_int = aov[aov["Source"] == "cond * bout"]
            pint = float(row_int[pcol].values[0]) if len(row_int) else np.nan
            pint_gg = float(row_int[gg].values[0]) if len(row_int) else np.nan
        except Exception:
            pint = pint_gg = np.nan

        rows.append(dict(feature=feat, n=n,
                         dz_b1=bout_dz[1], p_b1=bout_p[1],
                         dz_b2=bout_dz[2], p_b2=bout_p[2],
                         dz_b3=bout_dz[3], dz_b4=bout_dz[4],
                         slope_mean=slope_mean, trend_p=tr,
                         inter_p=pint, inter_p_gg=pint_gg))
    res = pd.DataFrame(rows)
    res["p_b1_fdr"] = bh_fdr(res["p_b1"].values)
    res.to_csv(OUT / "stats_bout.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    show = res.copy()
    for c in ["dz_b1", "dz_b2", "dz_b3", "dz_b4", "p_b1", "p_b1_fdr", "p_b2",
              "trend_p", "inter_p"]:
        show[c] = show[c].round(3)
    show["slope_mean"] = show["slope_mean"].apply(lambda v: f"{v:.2g}")
    print(show[["feature", "n", "dz_b1", "p_b1", "p_b1_fdr", "dz_b2", "dz_b3", "dz_b4",
                "slope_mean", "trend_p", "inter_p"]].to_string(index=False))
    print("\nbout1 significant (FDR<.05):", res.loc[res.p_b1_fdr < 0.05, "feature"].tolist())
    print("decay trend significant (p<.05):", res.loc[res.trend_p < 0.05, "feature"].tolist())
    print("interaction cond*bout (p<.05):", res.loc[res.inter_p < 0.05, "feature"].tolist())


if __name__ == "__main__":
    main()
