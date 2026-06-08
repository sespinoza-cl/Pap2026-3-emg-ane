"""Estadistica pareada Anestesia (A1) vs Placebo (A2) del EMG masticatorio.

Para cada metrica, sobre el lado elegido y sujetos incluidos:
  - t pareado y Wilcoxon signed-rank (intra-sujeto; corrige el error del
    analisis previo del lab que uso Mann-Whitney no pareado).
  - Tamano de efecto: Cohen dz (pareado) y r de rangos.
  - Equivalencia TOST con margen d=0.5 (efecto medio como SESOI).
  - Correccion FDR (Benjamini-Hochberg) de los p de Wilcoxon.
"""
import numpy as np
import pandas as pd
from scipy import stats
from config import OUT

FEATURES = ["MDF", "MNF", "peakF", "totpow", "rms_amp",
            "p_20_60", "p_60_150", "p_150_250", "p_250_450",
            "chew_rate", "cv_ipi", "med_dur", "med_amp", "med_area", "duty",
            "mdf_slope", "mnf_slope"]
SESOI_D = 0.5  # margen de equivalencia (Cohen d)


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    pp = p[idx]
    order = np.argsort(pp)
    n = len(pp)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        val = pp[i] * n / (rank + 1)
        prev = min(prev, val)
        adj[i] = prev
    out[idx] = adj
    return out


def tost_paired(diff, sesoi_d):
    """TOST de equivalencia sobre diferencias pareadas. Margen = sesoi_d * SD(diff)."""
    d = diff[np.isfinite(diff)]
    n = len(d)
    if n < 3:
        return np.nan, np.nan, np.nan
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    bound = sesoi_d * sd
    # H0a: mean <= -bound ; H0b: mean >= +bound
    t1 = (m - (-bound)) / se
    t2 = (m - bound) / se
    p1 = stats.t.sf(t1, n - 1)        # P(T > t1)  para limite inferior
    p2 = stats.t.cdf(t2, n - 1)       # P(T < t2)  para limite superior
    p_tost = max(p1, p2)
    return p_tost, -bound, bound


def main():
    df = pd.read_csv(OUT / "features_all.csv")
    qc = pd.read_csv(OUT / "qc_table.csv")
    inc = qc[qc["auto_ok"]]
    print(f"Sujetos incluidos (auto-OK): {len(inc)}")

    # seleccionar lado elegido por sujeto
    sel = inc[["subj", "side"]]
    d = df.merge(sel, on=["subj", "side"])

    rows = []
    for feat in FEATURES:
        wide = d.pivot_table(index="subj", columns="cond", values=feat)
        wide = wide.dropna(subset=["ANE", "PLA"])
        ane, pla = wide["ANE"].values, wide["PLA"].values
        diff = ane - pla
        n = len(diff)
        # tests
        tt = stats.ttest_rel(ane, pla)
        try:
            w = stats.wilcoxon(ane, pla)
            wp = w.pvalue
        except ValueError:
            wp = np.nan
        sh = stats.shapiro(diff).pvalue if n >= 3 else np.nan
        dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
        # r de rangos (matched-pairs rank-biserial)
        rb = np.nan
        nz = diff[diff != 0]
        if len(nz) > 0:
            ranks = stats.rankdata(np.abs(nz))
            rpos = ranks[nz > 0].sum()
            rneg = ranks[nz < 0].sum()
            T = rpos + rneg
            rb = (rpos - rneg) / T if T > 0 else np.nan
        p_tost, lo, hi = tost_paired(diff, SESOI_D)
        rows.append(dict(feature=feat, n=n,
                         mean_ANE=np.mean(ane), mean_PLA=np.mean(pla),
                         mean_diff=diff.mean(), sd_diff=diff.std(ddof=1),
                         shapiro_p=sh, t_p=tt.pvalue, wilcoxon_p=wp,
                         cohen_dz=dz, rank_biserial=rb,
                         tost_p=p_tost, equiv_lo=lo, equiv_hi=hi))
    res = pd.DataFrame(rows)
    res["wilcoxon_fdr"] = bh_fdr(res["wilcoxon_p"].values)
    res["sig_fdr"] = res["wilcoxon_fdr"] < 0.05
    res["equivalent"] = res["tost_p"] < 0.05
    res.to_csv(OUT / "stats_ANE_vs_PLA.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    show = res[["feature", "n", "mean_ANE", "mean_PLA", "mean_diff",
                "cohen_dz", "wilcoxon_p", "wilcoxon_fdr", "tost_p",
                "sig_fdr", "equivalent"]].copy()
    for c in ["mean_ANE", "mean_PLA", "mean_diff"]:
        show[c] = show[c].apply(lambda v: f"{v:.3g}")
    for c in ["cohen_dz", "wilcoxon_p", "wilcoxon_fdr", "tost_p"]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print("\nGuardado:", OUT / "stats_ANE_vs_PLA.csv")
    print("\nSignificativos (FDR<.05):", res.loc[res.sig_fdr, "feature"].tolist())
    print("Equivalentes (TOST p<.05):", res.loc[res.equivalent, "feature"].tolist())


if __name__ == "__main__":
    main()
