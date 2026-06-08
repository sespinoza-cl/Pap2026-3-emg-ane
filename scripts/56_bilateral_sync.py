"""Bilateral EMG synchrony: L vs R masseter envelope correlation.

Design context:
  - Anesthesia applied to LEFT gum -> reduced L somatosensory feedback
  - Subjects chewed on RIGHT side
  - BOTH masseters are bilaterally active (CPG-driven)

Metrics per subject x condition x bout:
  sync_r    : Pearson r(env_L, env_R) within bout
  asym      : (L_amp - R_amp) / (L_amp + R_amp)  bilateral amplitude balance
  amp_L     : mean RMS envelope L
  amp_R     : mean RMS envelope R
  lag_ms    : cross-correlation peak lag (positive = L leads R)

QC: both L AND R must have MDF in [45, 200] Hz and chew_rate in [0.4, 2.8] Hz.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import welch, find_peaks, correlate
from scipy.stats import wilcoxon
import pingouin as pg
from config import DERIV, OUT
from fig_style import RC, FIGSIZE, ANEC as ANE_C, PLAC as PLA_C, SUPPL, save_fig

BOUT_S = 60.0
FS     = 1024.0
ENV_WIN_MS = 50           # RMS envelope window


# ─────────────────────────────────────────────────────────────────────────────
# Signal utilities
# ─────────────────────────────────────────────────────────────────────────────

def rms_env(sig, fs=FS, win_ms=ENV_WIN_MS):
    w = max(1, int(round(win_ms * fs / 1000)))
    return np.sqrt(np.convolve(sig**2, np.ones(w) / w, mode="same"))


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
    pks, _ = find_peaks(z, height=np.percentile(z, 50),
                        distance=int(0.4 * fs))
    if len(pks) < 3:
        return np.nan
    ipi = np.diff(pks) / fs
    ipi = ipi[ipi <= 1.5]
    return float(1.0 / np.median(ipi)) if ipi.size else np.nan


def reshape_bouts(sig, n_bouts=4):
    L = int(round(BOUT_S * FS))
    if sig.size == n_bouts * L:
        return sig.reshape(n_bouts, L)
    nb = sig.size // L
    return sig[:nb * L].reshape(nb, L) if nb >= 1 else sig[None, :]


# ─────────────────────────────────────────────────────────────────────────────
# QC: check if a side is physiologically valid
# ─────────────────────────────────────────────────────────────────────────────

def side_ok(sig, fs=FS, mdf_lo=45, mdf_hi=200, rate_lo=0.4, rate_hi=2.8):
    mdf  = compute_mdf(sig, fs)
    rate = compute_chew_rate(sig, fs)
    if np.isnan(mdf) or np.isnan(rate):
        return False
    return (mdf_lo <= mdf <= mdf_hi) and (rate_lo <= rate <= rate_hi)


# ─────────────────────────────────────────────────────────────────────────────
# Per-bout bilateral metrics
# ─────────────────────────────────────────────────────────────────────────────

def bilateral_metrics(bout_l, bout_r):
    el = rms_env(np.abs(bout_l))
    er = rms_env(np.abs(bout_r))
    # Pearson correlation of envelopes
    r = float(np.corrcoef(el, er)[0, 1])
    # amplitude
    amp_l = float(el.mean())
    amp_r = float(er.mean())
    # bilateral asymmetry (positive = L > R)
    asym = (amp_l - amp_r) / (amp_l + amp_r + 1e-12)
    # cross-correlation peak lag (ms)
    cc = correlate(el - el.mean(), er - er.mean(), mode="full")
    lags = np.arange(-(len(el) - 1), len(el))
    max_lag_idx = np.argmax(cc)
    lag_ms = float(lags[max_lag_idx] / FS * 1000)
    return r, asym, amp_l, amp_r, lag_ms


# ─────────────────────────────────────────────────────────────────────────────
# Main: build bilateral features table
# ─────────────────────────────────────────────────────────────────────────────

def build_bilateral_table():
    qc = pd.read_csv(OUT / "qc_table.csv")
    rows = []
    n_ok = 0
    for _, q in qc.iterrows():
        subj = q["subj"]
        npz_path = DERIV / f"{subj}_emg.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        fs = float(d["fs"])

        # check all 4 combos: both sides must be valid in both conditions
        ok = True
        for cond in ("ANE", "PLA"):
            for side in ("L", "R"):
                key = f"{cond}_{side}"
                if key not in d:
                    ok = False; break
                if not side_ok(d[key], fs):
                    ok = False; break
            if not ok:
                break

        if not ok:
            continue
        n_ok += 1

        for cond in ("ANE", "PLA"):
            bouts_l = reshape_bouts(d[f"{cond}_L"])
            bouts_r = reshape_bouts(d[f"{cond}_R"])
            n_bouts = min(len(bouts_l), len(bouts_r), 4)
            for bi in range(n_bouts):
                r, asym, amp_l, amp_r, lag = bilateral_metrics(
                    bouts_l[bi], bouts_r[bi]
                )
                rows.append(dict(
                    subj=subj, cond=cond, bout=bi + 1,
                    sync_r=r, asym=asym,
                    amp_L=amp_l * 1e6, amp_R=amp_r * 1e6,
                    lag_ms=lag,
                ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bilateral_sync.csv", index=False)
    print(f"Bilateral QC: {n_ok} subjects with valid L+R signals")
    print(f"  (from {len(qc)} total in qc_table)")
    print(df.groupby(["cond", "bout"])[["sync_r", "asym"]].mean().round(3))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Statistics per bout
# ─────────────────────────────────────────────────────────────────────────────

def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]; prev = min(prev, p[i] * n / (rank + 1)); adj[i] = prev
    return adj


def run_stats(df, feat):
    rows = []
    for b in [1, 2, 3, 4]:
        a = df[(df.cond == "ANE") & (df.bout == b)].set_index("subj")[feat]
        p = df[(df.cond == "PLA") & (df.bout == b)].set_index("subj")[feat]
        common = a.dropna().index.intersection(p.dropna().index)
        a, p = a.loc[common].values, p.loc[common].values
        diff = a - p
        n = len(diff)
        try:
            wp = wilcoxon(a, p).pvalue
        except Exception:
            wp = np.nan
        dz = diff.mean() / (diff.std(ddof=1) + 1e-12)
        rows.append(dict(bout=b, n=n, dz=dz, p=wp))

    res = pd.DataFrame(rows)
    res["p_fdr"] = bh_fdr(res["p"].values)

    # RM-ANOVA interaction cond x bout
    long = df[["subj", "cond", "bout", feat]].rename(columns={feat: "y"})
    try:
        aov = pg.rm_anova(data=long, dv="y", within=["cond", "bout"],
                          subject="subj", detailed=True)
        pcol = "p_unc" if "p_unc" in aov.columns else "p-unc"
        gg   = "p_GG_corr" if "p_GG_corr" in aov.columns else pcol
        row  = aov[aov["Source"] == "cond * bout"]
        inter_p = float(row[gg].values[0]) if len(row) else np.nan
    except Exception:
        inter_p = np.nan

    return res, inter_p


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, stem):
    save_fig(fig, stem, SUPPL)


def _plot_mean_sem(ax, df, feat, ylim=None):
    """Plot mean +/- SEM with shaded band for ANE and PLA. No individual lines."""
    bouts = sorted(df.bout.unique())
    for cond, c in [("ANE", ANE_C), ("PLA", PLA_C)]:
        sub = df[df.cond == cond].groupby("bout")[feat]
        mu  = np.array([sub.mean()[b] for b in bouts])
        se  = np.array([sub.sem()[b]  for b in bouts])
        x   = np.array(bouts)
        ax.fill_between(x, mu - se, mu + se, color=c, alpha=0.20)
        ax.plot(x, mu, "o-", color=c, ms=5, lw=1.6,
                label="Anesthesia" if cond == "ANE" else "Placebo")
        ax.errorbar(x, mu, yerr=se, fmt="none", color=c,
                    capsize=3, elinewidth=0.8)
    ax.set_xticks(bouts)
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_bilateral(df, feat, ylabel, stem, ylim=None):
    plt.rcParams.update(RC)
    st, inter_p = run_stats(df, feat)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _plot_mean_sem(ax, df, feat, ylim=ylim)

    if feat == "asym":
        ax.axhline(0, color="#888888", lw=0.8, ls="--")

    ylo, yhi = ax.get_ylim()
    rng = yhi - ylo
    for _, row in st.iterrows():
        if row["p_fdr"] < 0.05:
            sig = "**" if row["p_fdr"] < 0.01 else "*"
            ax.text(row["bout"], yhi + rng * 0.02,
                    "{} dz={:+.2f}".format(sig, row["dz"]),
                    ha="center", va="bottom", fontsize=7)
    ax.set_ylim(ylo, yhi + rng * 0.14)

    ax.set_xlabel("Bout (~3, 7.5, 12, 17 min post-application)")
    ax.set_ylabel(ylabel)
    handles = [mpatches.Patch(color=ANE_C, label="Anesthesia"),
               mpatches.Patch(color=PLA_C, label="Placebo")]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    _save(fig, stem)
    return st, inter_p


def plot_bilateral_sides(df):
    """Three SEPARATE panels: asymmetry, L amplitude, R amplitude."""
    plt.rcParams.update(RC)

    def ylim_from_data(feat, pad=0.15):
        vals = df[feat].dropna()
        lo, hi = np.percentile(vals, 5), np.percentile(vals, 95)
        rng = hi - lo
        return (lo - pad * rng, hi + pad * rng)

    panels = [
        ("asym",  "Asymmetry (L-R)/(L+R)",  (-0.50, 0.12),  "s10_bilateral_asym"),
        ("amp_L", "Left amplitude (µV)",     ylim_from_data("amp_L"), "s10_bilateral_amp_L"),
        ("amp_R", "Right amplitude (µV)",    ylim_from_data("amp_R"), "s10_bilateral_amp_R"),
    ]
    for feat, ylabel, ylim, stem in panels:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        _plot_mean_sem(ax, df, feat, ylim=ylim)
        if feat == "asym":
            ax.axhline(0, color="#888888", lw=0.8, ls="--")
        ax.set_xlabel("Bout (~3, 7.5, 12, 17 min post-application)")
        ax.set_ylabel(ylabel)
        handles = [mpatches.Patch(color=ANE_C, label="Anesthesia"),
                   mpatches.Patch(color=PLA_C, label="Placebo")]
        ax.legend(handles=handles)
        fig.tight_layout()
        _save(fig, stem)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    df = build_bilateral_table()
    if df.empty or df.subj.nunique() < 5:
        print("Not enough bilateral subjects — check QC thresholds.")
        return

    # Print stats summaries
    for feat, label in [("sync_r", "Envelope correlation r(L,R)"),
                        ("asym",   "Amplitude asymmetry (L-R)/(L+R)")]:
        st, inter_p = run_stats(df, feat)
        print(f"\n--- {label}  (cond x bout interaction p={inter_p:.4f}) ---")
        print(st[["bout", "n", "dz", "p", "p_fdr"]].round(4).to_string(index=False))

    st.to_csv(OUT / "bilateral_stats.csv", index=False)

    # Figures
    plot_bilateral(df, "sync_r",
                   "Envelope correlation r(L, R)",
                   "s10_bilateral_sync",
                   ylim=(0.55, 1.02))

    plot_bilateral_sides(df)

    print("\nDone. Files: bilateral_sync.csv, bilateral_stats.csv, fig_bilateral_*.png")


if __name__ == "__main__":
    main()
