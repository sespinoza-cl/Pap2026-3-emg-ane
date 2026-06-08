# Manuscript plan — Transient masticatory sensorimotor compensation (Paper3)

## Central message (framing)
**Unilateral topical anesthesia induces a transient sensorimotor compensation of
chewing: greater masseter EMG amplitude and regularity during the first minute
after application, with unchanged frequency and rhythm, fading with the washout of
the anesthetic.**

Relationship to the parent paper (Sci Rep): the parent paper showed that chewing
improves cognition and increases frontocentral theta, with **no anesthesia effect**
on behavior or EEG, concluding a "central mechanism, independent of sensory
feedback". Paper3 **does not contradict but refines and strengthens** that conclusion:
1. **Validates the manipulation**: lidocaine DID have a physiological (peripheral,
   motor) effect, rebutting the criticism that "anesthesia did nothing".
2. **Explains the downstream null**: the motor system compensates (raises
   gain/amplitude) to **preserve the rhythmic CPG output**; with rhythm preserved,
   theta entrainment and the cognitive benefit are preserved → hence no behavioral/
   EEG anesthesia effect in the parent paper.
3. **Clarifies the mechanism**: trigeminal (periodontal/mucosal) afference modulates
   the GAIN of masseter drive, not the TIMING (set by the brainstem CPG).

## Results structure (proposed order)
1. **Sample and EMG quality.** N=34 (M3/M7 excluded for compromised electrodes);
   bipolar side selection on a physiological criterion (R=28/L=8). [QC supp fig]
2. **No global differences when pooling the session.** Median frequency, rate,
   duration, duty and fatigue (MDF/MNF slope) EQUIVALENT (TOST); only a weak
   amplitude trend. → replicates the parent-paper null at the global EMG level.
3. **Time-resolved transient effect (main finding).** At bout 1 (≈3 min
   post-application): cycle amplitude ↑ under anesthesia (med_amp dz=0.70, FDR=.003;
   RMS, 20–150 Hz power too), regularity ↑ (CV dz=−0.43, FDR=.042). **Condition×bout
   interaction** (med_amp p<.001). Washout: Δ decays 30→0 µV (trend p=.0002), 76% of
   subjects ANE>PLA at bout 1. [Fig 3]
4. **Cycle-locked time-frequency.** Significant cluster of greater broadband EMG
   power (~30–300 Hz) around the cycle peak at bout 1 (cluster p=.035, directional),
   absent at bout 4. [Fig 4]
5. **Amplitude vs timing dissociation.** Frequency/rate/duty equivalent while
   amplitude changes → signature of **gain compensation with intact timing**.

## Discussion structure (linking to the parent paper)
- **Sensorimotor compensation**: transient reduction of periodontal/mucosal afference
  by lidocaine (Na-channel block, does not cross the BBB) reduces force/contact
  feedback; the CNS responds by increasing masseter recruitment to maintain effective
  bite force. Bilateral trigeminal integration (fits chewing contralateral to anesthesia).
- **CPG robust in frequency**: the brainstem central pattern generator sets the
  rhythm (frequency/rate unchanged) while gain is adjusted by feedback → hierarchical
  control model (central timing, peripheral gain).
- **Link to the parent paper**: compensation preserves the rhythmic drive → preserves
  frontocentral theta entrainment and the cognitive benefit. Reinterprets the parent
  paper's anesthesia null as **motor homeostasis**, not absence of effect. Confirms the
  manipulation was effective (post-hoc manipulation check).
- **Causal time course**: the effect is maximal early and dissipates (~minutes),
  consistent with topical lidocaine pharmacokinetics → causal argument.
- **Limitations**: EMG amplitude as a proxy for effort (no force transducer);
  topical/unilateral anesthesia (not deep block); 76% (not universal); 4 bouts per
  condition; no comparison with full/injectable block.

## Figure plan
### Main
- **Fig 1 — Design and processing.** 2×2 schematic, masseter electrode montage
  (bipolar L/R), timeline (application → 4 bouts at ~3/7.5/12/17 min), pipeline
  (1024 Hz filter → Zapline → bipolar → cycles).
- **Fig 2 — No global effect / equivalence.** Effect-size forest with equivalence
  band (TOST) + paired boxplots (4-bout average) of key metrics.
  [`fig_forest_efectos.png`, `fig_paired_box.png`]
- **Fig 3 — Transient effect (KEY).** (a) by-bout boxplots ANE vs PLA (med_amp, RMS,
  CV); (b) washout curve with fit; (c) bout-1 individual consistency.
  [`fig_bout_box.png`, `fig_washout.png`]
- **Fig 4 — Cycle-locked time-frequency.** ANE, PLA and difference maps with the
  significant cluster; bout 1 vs bout 4.
  [`fig_cycle_tf.png`, `fig_cycle_tf_cluster.png`]

### Supplementary
- **S1** — QC: per-subject montage + SNR/side/inclusion table. [`fig_qc_montaje.png`, `qc_table.csv`]
- **S2** — Full equivalence (TOST) table and pooled statistics for all metrics. [`stats_ANE_vs_PLA.csv`]
- **S3** — Grand-average PSD ANE/PLA/Rest (selected side). [`fig_psd_grand.png`]
- **S4** — Full by-bout statistics table (Wilcoxon, trend, RM-ANOVA). [`stats_bout.csv`]
- **S5** — Robustness: results with N=30/32/36 and alternative side selection. [TODO]
- **S6** — Fatigue: MDF/MNF time course within the bout (no effect). [`fig_tf_mdf_timecourse.png`]
- **S7** — Full cycle-locked TF per condition and across the 4 bouts.
- **S8** — Order/carryover control. [`fig_order_carryover.png`]
- **S9** — Per-subject diagnostics (good/borderline examples). [`diag_<subj>.png`]

## Data provenance & IP disclosure (ready-to-paste draft text)
> `[REF_PAPER1]` = Espinoza S, Cáceres S, Salinas M, Moraga-Espinoza D,
> Morreal-Ortega L, El-Deredy W. "Chewing modulates theta oscillation and functional
> connectivity of the frontocentral cortex in attention and working memory."
> *Scientific Reports* [YEAR];[VOL]:[ART]. DOI: [INSERT]. (Sci Rep is CC BY 4.0;
> authors retain copyright.) Insert the data-repository DOI/accession if the dataset
> was deposited.

**Methods — Participants/Design (shared-dataset note).**
"The data analyzed here were acquired as part of a previously published study
([REF_PAPER1]) and are re-analyzed here for a different purpose. That study used a
2×2 within-subject design (chewing × dental anesthesia) and reported behavioral and
EEG outcomes; the present work is a secondary analysis focusing on the surface
masseter electromyogram (EMG), which was not analyzed previously. No new data were
collected. Behavioral measures from the parent study are used here only as contextual
covariates and are not re-reported as original findings."

**Transparency / overlap statement (cover letter or a footnote).**
"This manuscript uses the same participant sample and recording sessions as
[REF_PAPER1]. The two papers do not overlap in their primary outcomes: the parent
paper addressed chewing-related cognitive and theta-band effects, whereas the present
paper addresses the masticatory EMG response to topical anesthesia. Any behavioral or
EEG values reported here are cited to [REF_PAPER1]."

**Data Availability statement.**
"The dataset supporting this study is the same as that of [REF_PAPER1] and is
available at [REPOSITORY/DOI]. Derived EMG measures and analysis code for the present
study are available at [REPO/DOI for Paper3 code]."

**Ethics statement.**
"All procedures were approved by the Ethics Committee of the School of Dentistry,
Universidad de Valparaíso (registration code POSTG-06-22), and conducted in accordance
with the Declaration of Helsinki. Participants provided written informed consent,
which covered the use of their data for research; the present secondary analysis falls
within the scope of that approval and consent."

**Copyright note (internal, not for the manuscript).** The parent paper is open access
under CC BY 4.0, so the authors retain copyright; reusing the data and (with citation)
published material is permitted. Reproducing parent-paper figures/text verbatim would
require citation/attribution under CC BY. Self-citation + a shared-dataset declaration
address redundant-publication/self-plagiarism concerns. Confirm journal policy and
institutional research-office guidance before submission.

## Pending to close the analysis
- Robustness N=30/32/36 (S5) and alternative inclusion threshold.
- Correlation of the compensatory effect (bout-1 Δ) with the parent paper's
  behavioral/theta metrics (would directly link the two stories).
- Decide the primary pre-registrable metric (med_amp); report the rest as exploratory.
