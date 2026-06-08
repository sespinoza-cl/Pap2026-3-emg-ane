"""Robustness of the bout-1 effect across inclusion cohorts (N=30/32/34/36).

Recomputes per-subject median cycle amplitude (bout 1 and 4-bout average) from the
.npz, on the physiologically selected side (qc_table), and runs the paired
Anesthesia vs Placebo test for each cohort. Also the cond x bout RM-ANOVA interaction.
"""
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from config import OUT, DERIV, SUBJECTS_36, LISTA30, LISTA32
from emg_features import all_features
from emg_tf import reshape_bouts

FEAT = "med_amp"
SCALE = 1e6  # uV


def per_subject_table():
    qc = pd.read_csv(OUT / "qc_table.csv").set_index("subj")
    rows = []
    for subj in SUBJECTS_36:
        p = DERIV / f"{subj}_emg.npz"
        if not p.exists() or subj not in qc.index:
            continue
        side = qc.loc[subj, "side"]
        d = np.load(p); fs = float(d["fs"]); rs = d[f"RS_{side}"]
        rec = {"subj": subj, "auto_ok": bool(qc.loc[subj, "auto_ok"])}
        for cond in ("ANE", "PLA"):
            bouts = reshape_bouts(d[f"{cond}_{side}"], fs, n_bouts=4)
            amps = [all_features(b, fs, rs_sig=rs)[FEAT] for b in bouts]
            rec[f"{cond}_b1"] = amps[0] * SCALE
            rec[f"{cond}_avg"] = np.nanmean(amps) * SCALE
            for i, v in enumerate(amps, 1):
                rec[f"{cond}_b{i}"] = v * SCALE
        rows.append(rec)
    return pd.DataFrame(rows)


def paired(a, p):
    a, p = np.asarray(a), np.asarray(p)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    diff = a - p
    dz = diff.mean() / diff.std(ddof=1)
    try:
        wp = stats.wilcoxon(a, p).pvalue
    except ValueError:
        wp = np.nan
    return len(a), diff.mean(), dz, wp, 100*np.mean(diff > 0)


def interaction_p(df_sub):
    long = []
    for _, r in df_sub.iterrows():
        for cond in ("ANE", "PLA"):
            for b in (1, 2, 3, 4):
                long.append((r["subj"], cond, b, r[f"{cond}_b{b}"]))
    long = pd.DataFrame(long, columns=["subj", "cond", "bout", "y"]).dropna()
    try:
        aov = pg.rm_anova(data=long, dv="y", within=["cond", "bout"], subject="subj")
        pcol = "p_unc" if "p_unc" in aov.columns else "p-unc"
        return float(aov.loc[aov["Source"] == "cond * bout", pcol].values[0])
    except Exception:
        return np.nan


def main():
    T = per_subject_table()
    cohorts = {
        "QC auto-OK (N=34)": T[T.auto_ok]["subj"].tolist(),
        "lista30": [s for s in LISTA30 if s in set(T.subj)],
        "lista32": [s for s in LISTA32 if s in set(T.subj)],
        "lista36 (all)": T["subj"].tolist(),
    }
    print(f"{'cohort':<20}{'N':>4} | bout1: dz   p_wilcox  %pos | avg: dz   p     | interaction p")
    print("-" * 90)
    for name, subs in cohorts.items():
        sub = T[T.subj.isin(subs)]
        n1, m1, dz1, p1, pos1 = paired(sub["ANE_b1"], sub["PLA_b1"])
        na, ma, dza, pa, posa = paired(sub["ANE_avg"], sub["PLA_avg"])
        pint = interaction_p(sub)
        print(f"{name:<20}{n1:>4} | {dz1:>5.2f} {p1:>8.4f} {pos1:>5.0f}% | "
              f"{dza:>5.2f} {pa:>6.3f} | {pint:.4f}")
    T.to_csv(OUT / "robustness_table.csv", index=False)
    print("\nSaved:", OUT / "robustness_table.csv")


if __name__ == "__main__":
    main()
