"""Laterality analysis of the anesthesia EMG effect.

Design (fixed across subjects):
  - Anesthesia applied to the LEFT gum  -> reduced LEFT mucosal afference
  - Subjects instructed to chew on the RIGHT side (working side = right)
  - Both masseters are bilaterally active (single U-shaped mandible, CPG-driven)

Questions:
  A. Compliance / working side: is the RIGHT masseter the higher-amplitude
     (working) side, as instructed? (amp_R > amp_L  ->  asym < 0)
  B. Effect laterality: does the bout-1 anesthesia amplitude increase appear on
     the RIGHT (working / contralateral-to-anesthesia), the LEFT
     (balancing / ipsilateral-to-anesthesia), or both?
  C. Within-subject lateralisation: is the ANE-PLA amplitude change larger on R
     than on L? (tests whether the effect is non-local to the anesthetised side)
  D. Selected-side subgroup: does the effect track the side selected in the main
     analysis (mostly R)?

Reuses the exact QC and amplitude definitions from 56_bilateral_sync.py
(RMS-envelope amplitude per bout per side, side_ok physiological QC).
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
from scipy.signal import welch, find_peaks
from scipy.stats import wilcoxon
from config import DERIV, OUT
from emg_features import cycle_features
from fig_style import RC, FIGSIZE, ANEC, MAIN, save_fig

FS = 1024.0
ENV_WIN_MS = 50
BOUT_S = 60.0


# ---- exact signal utilities (copied from 56_bilateral_sync.py) ----
def rms_env(sig, fs=FS, win_ms=ENV_WIN_MS):
    w = max(1, int(round(win_ms * fs / 1000)))
    return np.sqrt(np.convolve(sig ** 2, np.ones(w) / w, mode="same"))


def compute_mdf(sig, fs, fmin=20, fmax=450):
    if sig.size < fs:
        return np.nan
    f, pxx = welch(sig, fs=fs, nperseg=int(fs))
    m = (f >= fmin) & (f <= fmax)
    f, pxx = f[m], pxx[m]
    tot = np.trapz(pxx, f)
    if tot <= 0:
        return np.nan
    cum = np.cumsum((pxx[:-1] + pxx[1:]) / 2 * np.diff(f))
    idx = np.searchsorted(cum, tot / 2)
    return float(f[min(idx + 1, len(f) - 1)])


def compute_chew_rate(sig, fs):
    env = rms_env(np.abs(sig), fs)
    mu, sd = env.mean(), env.std() + 1e-12
    z = (env - mu) / sd
    pks, _ = find_peaks(z, height=np.percentile(z, 50), distance=int(0.4 * fs))
    if len(pks) < 3:
        return np.nan
    ipi = np.diff(pks) / fs
    ipi = ipi[ipi <= 1.5]
    return float(1.0 / np.median(ipi)) if ipi.size else np.nan


def side_ok(sig, fs=FS, mdf_lo=45, mdf_hi=200, rate_lo=0.4, rate_hi=2.8):
    mdf = compute_mdf(sig, fs)
    rate = compute_chew_rate(sig, fs)
    if np.isnan(mdf) or np.isnan(rate):
        return False
    return (mdf_lo <= mdf <= mdf_hi) and (rate_lo <= rate <= rate_hi)


def reshape_bouts(sig, n_bouts=4):
    L = int(round(BOUT_S * FS))
    if sig.size == n_bouts * L:
        return sig.reshape(n_bouts, L)
    nb = sig.size // L
    return sig[:nb * L].reshape(nb, L) if nb >= 1 else sig[None, :]


def amp_bout(bout):
    return float(rms_env(np.abs(bout)).mean()) * 1e6  # microV


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]; prev = min(prev, p[i] * n / (rank + 1)); adj[i] = prev
    return adj


def paired(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    diff = a - p
    n = len(diff)
    try:
        wp = wilcoxon(a, p).pvalue
    except Exception:
        wp = np.nan
    dz = diff.mean() / (diff.std(ddof=1) + 1e-12)
    return n, dz, float(diff.mean()), wp


# ---- build per subject x cond x bout table (bilateral-valid subset) ----
def build():
    qc = pd.read_csv(OUT / "qc_table.csv")
    sel_side = dict(zip(qc["subj"], qc["side"]))
    rows = []
    n_ok = 0
    for subj in qc["subj"]:
        npz = DERIV / f"{subj}_emg.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        fs = float(d["fs"])
        ok = True
        for cond in ("ANE", "PLA"):
            for side in ("L", "R"):
                k = f"{cond}_{side}"
                if k not in d or not side_ok(d[k], fs):
                    ok = False; break
            if not ok:
                break
        if not ok:
            continue
        n_ok += 1
        rsL, rsR = d["RS_L"], d["RS_R"]
        for cond in ("ANE", "PLA"):
            bl = reshape_bouts(d[f"{cond}_L"]); br = reshape_bouts(d[f"{cond}_R"])
            nb = min(len(bl), len(br), 4)
            for bi in range(nb):
                aL, aR = amp_bout(bl[bi]), amp_bout(br[bi])
                mL = cycle_features(bl[bi], fs, rs_sig=rsL).get("med_amp", np.nan)
                mR = cycle_features(br[bi], fs, rs_sig=rsR).get("med_amp", np.nan)
                rows.append(dict(subj=subj, sel_side=sel_side.get(subj, "?"),
                                 cond=cond, bout=bi + 1, amp_L=aL, amp_R=aR,
                                 mamp_L=float(mL) * 1e6, mamp_R=float(mR) * 1e6,
                                 asym=(aL - aR) / (aL + aR + 1e-12)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "laterality_table.csv", index=False)
    print(f"Bilateral-valid subjects: {n_ok}\n")
    return df


def wide(df, cond, bout, col):
    return df[(df.cond == cond) & (df.bout == bout)].set_index("subj")[col]


def make_figures(df):
    """Assertion-evidence panels (FIGSIZE each, collage-ready, no titles).
    Numbers live in the Results text; figures carry only significance marks."""
    plt.rcParams.update(RC)
    GREY = "#9A9A9A"

    def stars(p):
        return ("***" if p < 0.001 else "**" if p < 0.01
                else "*" if p < 0.05 else "n.s.")

    def diff_side(side):
        a = wide(df, "ANE", 1, f"mamp_{side}")
        p = wide(df, "PLA", 1, f"mamp_{side}")
        idx = a.dropna().index.intersection(p.dropna().index)
        return a.loc[idx] - p.loc[idx]

    dR, dL = diff_side("R"), diff_side("L")
    common = dR.index.intersection(dL.index)
    dRv, dLv = dR.loc[common].values, dL.loc[common].values
    _, _, _, pR = paired(wide(df, "ANE", 1, "mamp_R").loc[common].values,
                         wide(df, "PLA", 1, "mamp_R").loc[common].values)
    _, _, _, pL = paired(wide(df, "ANE", 1, "mamp_L").loc[common].values,
                         wide(df, "PLA", 1, "mamp_L").loc[common].values)
    _, _, _, pC = paired(dRv, dLv)

    # Panel A: lateralisation of the bout-1 effect (working vs balancing)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    data, pos, cols = [dRv, dLv], [1, 2], [ANEC, GREY]
    bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color="k", lw=1.4),
                    whiskerprops=dict(color="#555"), capprops=dict(color="#555"),
                    boxprops=dict(color="#555"))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.30)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        x = pos[i] + rng.uniform(-0.10, 0.10, len(d))
        ax.scatter(x, d, s=18, color=cols[i], edgecolor="white",
                   linewidth=0.4, zorder=3, alpha=0.95)
    ax.axhline(0, color="#444", lw=0.8, ls="--")
    ax.set_xticks(pos)
    ax.set_xticklabels(["Working\n(right)", "Balancing\n(left)"])
    ax.set_ylabel(r"$\Delta$ median cycle amplitude ($\mu$V)")
    yhi = max(d.max() for d in data); ylo = min(d.min() for d in data)
    r2 = yhi - ylo
    # per-box significance vs zero
    ax.text(1, yhi + r2 * 0.03, stars(pR), ha="center", va="bottom", fontsize=12)
    ax.text(2, yhi + r2 * 0.03, stars(pL), ha="center", va="bottom", fontsize=9)
    # comparison bracket (working vs balancing)
    ybar, h = yhi + r2 * 0.20, r2 * 0.03
    ax.plot([1, 1, 2, 2], [ybar, ybar + h, ybar + h, ybar], color="#333", lw=1.0)
    ax.text(1.5, ybar + h, stars(pC), ha="center", va="bottom", fontsize=12)
    ax.set_ylim(ylo - r2 * 0.08, ybar + r2 * 0.16)
    fig.tight_layout()
    save_fig(fig, "s11_laterality_bout1", MAIN)
    # (working-side compliance reported as text in Results, not as a panel)


def main():
    df = build()
    subs = sorted(df.subj.unique())

    # ---------- A. Working side / compliance ----------
    print("=== A. Working side (compliance with instructed RIGHT chewing) ===")
    print("    asym = (L-R)/(L+R);  asym<0 => RIGHT-dominant (working side = right)")
    for b in [1, 2, 3, 4]:
        a = df[(df.bout == b)].groupby("subj")["asym"].mean()
        frac_R = float((a < 0).mean())
        print(f"  bout {b}: mean asym={a.mean():+.3f}  | R-dominant subjects: "
              f"{frac_R*100:.0f}%  ({int((a<0).sum())}/{len(a)})")
    a_all = df.groupby("subj")["asym"].mean()
    print(f"  OVERALL: {int((a_all<0).sum())}/{len(a_all)} subjects R-dominant "
          f"({(a_all<0).mean()*100:.0f}%); mean asym={a_all.mean():+.3f}\n")

    metrics = [("amp", "RMS amplitude (whole-bout)"),
               ("mamp", "median cycle amplitude (per-chew; = MAIN metric, dz=0.70)")]

    # ---------- B. Effect laterality: ANE vs PLA on each side, per bout ----------
    for m, mlabel in metrics:
        print(f"\n=== B[{m}]. Anesthesia effect by side -- {mlabel} ===")
        for side, slab in [("R", "RIGHT (working / contralateral-to-anesthesia)"),
                           ("L", "LEFT  (balancing / ipsilateral-to-anesthesia)")]:
            col = f"{m}_{side}"
            print(f"  -- {col}: {slab} --")
            recs, ps = [], []
            for b in [1, 2, 3, 4]:
                aa, pp = wide(df, "ANE", b, col), wide(df, "PLA", b, col)
                idx = aa.dropna().index.intersection(pp.dropna().index)
                n, dz, dmu, wp = paired(aa.loc[idx].values, pp.loc[idx].values)
                recs.append((b, n, dz, dmu, wp)); ps.append(wp)
            fdr = bh_fdr(ps)
            for (b, n, dz, dmu, wp), q in zip(recs, fdr):
                star = " *" if q < 0.05 else ""
                print(f"    bout {b}: n={n}  dz={dz:+.2f}  d={dmu:+.1f}uV  "
                      f"p={wp:.4f}  FDR={q:.4f}{star}")

    # ---------- C. Within-subject lateralisation at bout 1 (dR vs dL) ----------
    print("\n=== C. Within-subject lateralisation at bout 1 (ANE-PLA: R vs L) ===")
    for m, mlabel in metrics:
        aR1, pR1 = wide(df, "ANE", 1, f"{m}_R"), wide(df, "PLA", 1, f"{m}_R")
        aL1, pL1 = wide(df, "ANE", 1, f"{m}_L"), wide(df, "PLA", 1, f"{m}_L")
        idx = (aR1.dropna().index.intersection(pR1.dropna().index)
               .intersection(aL1.dropna().index).intersection(pL1.dropna().index))
        dR = (aR1.loc[idx] - pR1.loc[idx]).values
        dL = (aL1.loc[idx] - pL1.loc[idx]).values
        n, dz, dmu, wp = paired(dR, dL)
        print(f"  [{m}] dR={dR.mean():+.1f} dL={dL.mean():+.1f}uV | "
              f"dR>dL: n={n} dz={dz:+.2f} p={wp:.4f} | "
              f"subj dR>dL: {int((dR>dL).sum())}/{len(dR)}")

    # ---------- D. Selected-side subgroup (median cycle amplitude) ----------
    print("\n=== D. Bout-1 effect on SELECTED side, split by selected side (median cycle amp) ===")
    for sside in ("R", "L"):
        subj_s = [s for s in subs if df[df.subj == s]["sel_side"].iloc[0] == sside]
        if not subj_s:
            continue
        col = f"mamp_{sside}"
        aa = wide(df[df.subj.isin(subj_s)], "ANE", 1, col)
        pp = wide(df[df.subj.isin(subj_s)], "PLA", 1, col)
        idx = aa.dropna().index.intersection(pp.dropna().index)
        if len(idx) >= 3:
            n, dz, dmu, wp = paired(aa.loc[idx].values, pp.loc[idx].values)
            print(f"  selected={sside} (n={n}): bout-1 ANE-PLA on {col}  "
                  f"dz={dz:+.2f}  d={dmu:+.1f}uV  p={wp:.4f}")
        else:
            print(f"  selected={sside}: n={len(idx)} too few")

    make_figures(df)
    print("\nFigure -> outputs/main/: s11_laterality_bout1.png")


if __name__ == "__main__":
    main()
