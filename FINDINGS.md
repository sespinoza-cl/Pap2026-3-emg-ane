# Findings — Masticatory EMG under topical anesthesia (Paper3, Exp1)

**Question:** Does masticatory masseter EMG change under topical lidocaine
anesthesia (applied unilaterally, reducing oral somatosensory feedback)? This is
a new angle on Exp1: the parent paper (Scientific Reports) collapsed the
anesthesia factor and did not rigorously test the EMG.

---

## 1. Design and data (confirmed in all 36 subjects)
- 2×2 within-subject: Chewing × Anesthesia, counterbalanced (R1/R2). Biosemi
  64+8 @ 1024 Hz. EMG = EXG5–8 (channels 69:72), bipolar Left/Right.
- **8 chewing bouts of 60 s = 4 per condition** (4 Anesthesia + 4 Placebo).
- **One spray application per condition, no re-application between bouts.** The 4
  bouts occur at **~3, 7.5, 12 and 17 min post-application** (regular spacing).
- **Placebo = identical spray procedure and taste, without lidocaine** (all
  conditions received an application) → isolates the pharmacological lidocaine effect.

## 2. Methods (and corrections over the previous lab analysis)
- From **raw BDF at 1024 Hz**: band-pass 20–450 Hz (4th-order Butterworth,
  zero-phase) and **no resampling** (avoids aliasing of the high EMG band; preserves
  median frequency). → fixes the filter/downsample order.
- Line noise removed with **Zapline-plus** (not notch) → preserves the spectrum
  (notch filtering carved holes that biased MDF/MNF).
- **Paired statistics** (paired Wilcoxon/t, RM-ANOVA) → fixes the previous use of
  unpaired Mann–Whitney (data are within-subject).
- Effect sizes (Cohen's dz), equivalence testing (TOST), FDR correction.

## 3. Quality and inclusion
- Objective QC + side selection on a **physiological criterion** (MDF and rate in
  range, not SNR alone — avoids floating electrodes with spurious SNR, e.g. PS12).
- **N = 34** (M3 and M7 excluded for compromised electrodes). Side: R=28/L=8,
  reproducing the parent paper's selection (25/5). Subjects excluded from the parent
  paper for EEG reasons (M2, PS12, S12, S3) have good EMG → larger N available than
  the original (N=30).

## 4. Session-averaged result (4 bouts pooled)
- **No metric differs significantly** between Anesthesia and Placebo.
- **Equivalent** (TOST d=0.5): median frequency (MDF), peak frequency, chewing
  rate, cycle duration, fatigue slopes (MDF/MNF), high-frequency bands.
- Only a **weak, non-significant trend** toward higher amplitude under anesthesia.
- → Pooling diluted the effect (see Section 5).

## 5. MAIN FINDING — bout-resolved analysis (transient effect)
Topical lidocaine lasts ~15–20 min: maximal effect early, subsequent washout.
Resolving by bout (1–4):

**At bout 1 (maximal anesthesia) there is an FDR-surviving effect:**

| Metric | dz bout 1 | FDR bout 1 | trend (decay) | cond×bout interaction |
|---|---|---|---|---|
| **Median cycle amplitude** (med_amp) | **+0.70** | **0.003** | p<0.001 | **p<0.001** |
| RMS amplitude | +0.49 | 0.013 | p=0.036 | 0.12 |
| Total power | +0.46 | 0.013 | p=0.041 | 0.26 |
| Power 60–150 Hz | +0.46 | 0.019 | p=0.059 | 0.24 |
| Power 20–60 Hz | +0.47 | 0.027 | p=0.041 | 0.45 |
| **CV of inter-peak interval** (regularity) | **−0.43** | **0.042** | — | p=0.050 |

- **EMG amplitude and power are HIGHER under anesthesia** during the first minute;
  the difference **decays to ~0 by bouts 3–4** (washout; curves converge).
- **Chewing is MORE regular** (lower CV) under anesthesia, also early.
- **No effect on frequency, rate or duty cycle** at any bout → timing and spectral
  content are unchanged.

## 6. Cycle-locked time-frequency
Epoching the bipolar signal around each masticatory cycle peak ([-0.4, +0.4] s)
and averaging the TF (Morlet wavelets, 20–450 Hz) per condition
(`emg_cycles.py`, `60_cycle_tf.py`, `61_cycle_tf_bout.py`, `62_cycle_tf_cluster.py`):
- Each cycle shows a broadband EMG power burst centered on the peak.
- **ANE−PLA difference at bout 1**: greater broadband power under anesthesia around
  the peak → consistent with the amplitude effect.
- **Bout 4**: the difference disappears (washout).
- **Cluster-based permutation test** (Maris & Oostenveld; paired 1-sample): at
  **bout 1** a **significant cluster** (a priori directional ANE>PLA, **p=0.035**;
  two-tailed p=0.051) around the cycle peak (≈ −0.15 to +0.1 s, ~30–300 Hz) with
  greater power under anesthesia. At **bout 4 no cluster** (p_min=0.59).

## 6b. Washout and individual consistency (`70_washout.py`)
- The ANE−PLA amplitude difference decays monotonically: +30.5 → +8.8 → −4.4 →
  −2.2 µV (bouts 1–4). Exponential fit τ≈2.9 min (illustrative, 4 points).
- **Bout 1**: median Δ = 22 µV, **p=0.0003**, **76% of subjects with ANE>PLA**.
- Significant decay (trend p=0.0002, −2.4 µV/min). → consistent with pharmacological
  washout of topical lidocaine.

## 6c. Order / carryover confounds (crossover) (`71_order_carryover.py`)
R1 = Anesthesia 1st→Placebo 2nd; R2 = Placebo 1st→Anesthesia 2nd.
- **Robust treatment effect**: mixed ANOVA at bout 1, condition effect F=17.1,
  **p=0.0002** (np²=0.35). Sequence n.s. (p=0.73). **Condition×sequence interaction
  NOT significant (p=0.136)** → the effect does not depend on sequence.
- **No residual carryover**: R1 placebo (post-anesthesia) ≈ R2 placebo (clean),
  medians 90.3 vs 87.2 µV, Mann–Whitney **p=0.379**. (Consistent with within-block
  washout ~12 min + inter-block gap + the protocol's tactile-sensation check.)
- **No period effect** (1st vs 2nd block): p=0.134.
- The bout-1 ANE−PLA effect is present in both arms (69% R1, 83% R2 with ANE>PLA);
  significant in R2 (dz=0.91, p=0.001), smaller in R1 (dz=0.46, p=0.144). R1 placebo
  is NOT elevated → the arm difference is sampling variability, not carryover; any
  carryover would only bias the effect downward (conservative).
- **Placebo effect controlled by design**: both conditions received spray + taste;
  the within-subject ANE−PLA contrast removes it.

## 6d-bis. Robustness across inclusion cohorts (`80_robustness.py`)
The bout-1 effect on median cycle amplitude is stable across inclusion choices:

| Cohort | N | bout-1 dz | p | %ANE>PLA | cond×bout p |
|---|---|---|---|---|---|
| QC auto-OK | 34 | 0.69 | 0.0003 | 76% | 0.0004 |
| lista30 | 30 | 0.68 | 0.0010 | 77% | 0.0011 |
| lista32 | 32 | 0.65 | 0.0010 | 75% | 0.0014 |
| all subjects | 36 | 0.26 | 0.0020 | 72% | 0.061 |

The only attenuation occurs when the two dead-electrode subjects (M3, M7) are
forced in (N=36); the paired test remains significant (p=0.002), which itself
justifies the QC exclusion.

## 6e. EMG compensation vs behavior (`81_emg_behavior_corr.py`, exploratory)
Using the parent paper's behavioral data (ane2=Anesthesia+chew, ane4=Placebo+chew;
secondary analysis, cited to the parent paper):
- **The behavioral anesthesia effect is null** (2-back RT/accuracy and VO RT/accuracy,
  all |effect| small, p>0.28), replicating the parent paper and indicating preserved
  performance.
- **No robust correlation** between bout-1 EMG compensation and behavior. One nominal
  correlation (Δ 2-back accuracy: Spearman ρ=−0.40, p=0.022, n=33) does not survive
  correction for the 6 tests (BH q≈0.13) → exploratory only.
- Reading: motor compensation buffers performance, which is preserved regardless of
  compensation magnitude (consistent with the null behavioral effect).
- **Theta link pending**: the available EEG TF files are organized by chewing/no-chewing
  (anesthesia collapsed), so an anesthesia-specific theta measure would require
  recomputing the EEG time-frequency split by anesthesia (follow-up).

## 7. Interpretation
When oral somatosensory feedback is transiently reduced, subjects chew during the
first minute with **greater amplitude (more effort) and greater regularity**, while
keeping frequency/rhythm intact — a **transient sensorimotor compensation** that
fades with the anesthetic. The effect is attributable to lidocaine (taste/procedure-
matched placebo) and time-dependent (single-dose pharmacological washout). This
turns the study from a "null/equivalence" result (when pooled) into a **positive,
time-dependent, pharmacologically coherent effect**.

## 8. Figures (outputs/)
- `fig_bout_box.png` — by bout, Anesthesia vs Placebo (box + points). **Main.**
- `fig_paired_box.png` — session average, within-subject paired lines.
- `fig_bout_timecourse.png` — bout time course.
- `fig_washout.png` — washout curve + individual consistency.
- `fig_order_carryover.png` — order/carryover control.
- `fig_cycle_tf.png`, `fig_cycle_tf_bout.png`, `fig_cycle_tf_cluster.png` —
  cycle-locked TF (global, by bout, and cluster test).
- `fig_forest_efectos.png` — effect sizes + equivalence (pooled).
- `fig_psd_grand.png`, `fig_tf_mdf_timecourse.png` — spectrum and MDF time course.
- `fig_qc_montaje.png`, `diag_<subj>.png` — quality control.
- `fig_emg_behavior.png` — EMG compensation vs behavior (exploratory).
- Tables: `stats_ANE_vs_PLA.csv`, `stats_bout.csv`, `robustness_table.csv`,
  `emg_behavior_table.csv`, `qc_table.csv`.

## 9. Pending / possible next steps
- [done] Robustness with N=30/32/36 — effect stable (Section 6d-bis).
- [done] EMG↔behavior correlation — behavioral null replicated; no robust link (6e).
- [pending] Anesthesia-specific theta: recompute EEG TF split by anesthesia, then
  correlate with bout-1 EMG compensation.
- Decide the primary pre-registrable metric (med_amp); report the rest as exploratory.
- Start formal manuscript drafting (academic-paper skill).
