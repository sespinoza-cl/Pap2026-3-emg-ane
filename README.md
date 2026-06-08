# Sensorimotor gain compensation during masticatory EMG under topical dental anaesthesia: a bout-resolved secondary analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Authors

| # | Author | Affiliation | ORCID |
|---|--------|-------------|-------|
| 1 | **Sebastian Espinoza** *(corresponding)* | Instituto de Tecnología para la Innovación en Salud y Bienestar, Faculty of Engineering, Andrés Bello National University, Chile; Dentistry School, Universidad de Valparaíso, Chile | [0000-0001-9678-2665](https://orcid.org/0000-0001-9678-2665) |
| 2 | **Klaus Samson** | Centro de Formación, Destreza e Innovación NOXIS, Clínica MEDs, Santiago, Chile | [0009-0007-1736-626X](https://orcid.org/0009-0007-1736-626X) |
| 3 | **Sebastián Jiménez** | Grupo de Investigación en Salud, Funcionalidad y Actividad Física (GISFAF), Kinesiología, Facultad de Ciencias de la Salud, Universidad Autónoma de Chile, Santiago, Chile; Brain Dynamics Laboratory, Universidad de Valparaíso, Chile | [0009-0005-8128-3579](https://orcid.org/0009-0005-8128-3579) |
| 4 | **Wael El-Deredy** | Brain Dynamics Lab, Interdisciplinary Center of Biomedical and Engineering Research for Health, Universidad de Valparaíso, Chile | [0000-0002-9822-1092](https://orcid.org/0000-0002-9822-1092) |

> Correspondence: inv.itisb@unab.cl

---

## Overview

This repository contains the analysis code, pre-computed outputs, and manuscript source (LaTeX) for a secondary EMG analysis of a within-subject crossover dataset examining sensorimotor adaptation during mastication under topical dental anaesthesia.

**Main finding:** Unilateral topical lidocaine (10%) increases masseter EMG amplitude and regularity at the first chewing bout post-application (*d*z = 0.695, FDR *p* = 0.004), with monotonic decay matching lidocaine pharmacokinetics, while chewing frequency and rhythm remain unchanged — consistent with sensorimotor gain compensation mediated by the brainstem central pattern generator.

---

## Repository structure

```
Paper3/
├── scripts/                    # Python analysis pipeline (run in numeric order)
│   ├── config.py               # Paths and global parameters
│   ├── emg_io.py               # BDF loading utilities
│   ├── emg_features.py         # Feature extraction (spectral + cycle)
│   ├── emg_cycles.py           # Cycle detection and TF epoching
│   ├── emg_tf.py               # Time-frequency (Morlet, STFT, fatigue slope)
│   ├── 10_preprocess.py        # Raw BDF → cleaned bipolar EMG (.npz)
│   ├── 22_bout_features.py     # Per-bout feature extraction
│   ├── 30_qc_select.py         # Quality control and participant selection
│   ├── 35_qc_figures.py        # QC diagnostic figures (S1, S9)
│   ├── 40_stats.py             # Session-averaged statistics + TOST
│   ├── 42_bout_stats.py        # Bout-resolved Wilcoxon + RM-ANOVA
│   ├── 52_bout_figures.py      # Main result figures (Fig 2)
│   ├── 56_bilateral_sync.py    # Bilateral synchrony analysis (S12)
│   ├── 60_cycle_tf.py          # Cycle-locked TF grand averages
│   ├── 62_cycle_tf_cluster.py  # Cluster-based permutation test (Fig 4)
│   ├── 70_washout.py           # Washout time course (Fig 3)
│   ├── 71_order_carryover.py   # Crossover confound analysis (S7)
│   ├── 80_robustness.py        # Robustness across inclusion thresholds (S8)
│   ├── 90_replot_figures.py    # Unified figure regeneration
│   └── 95_s5_stratified.py     # Within-bout fatigue slope figure (S5)
│
├── outputs/                    # Pre-computed results (all plots reproducible from here)
│   ├── bout_features.csv       # Per-subject × bout × condition EMG features
│   ├── stats_bout.csv          # Wilcoxon + dz statistics by bout
│   ├── qc_table.csv            # Participant inclusion decisions
│   ├── robustness_table.csv    # Sensitivity analysis across N
│   ├── bilateral_stats.csv     # Bilateral synchrony statistics
│   └── *.png                   # All manuscript figures (main + supplementary)
│
├── overleaf/                   # LaTeX manuscript source
│   ├── main.tex                # Master file
│   ├── text_parts/             # Abstract, Intro, Results, Discussion, Methods, Figs, Suppl
│   ├── figures/                # Figures (prefixed: fig2_, fig3_, fig4_, s1_, …, s12_)
│   ├── biblio.bib              # References
│   └── FIGURES_INDEX.txt       # Maps filenames ↔ figure numbers
│
├── .gitignore
└── README.md
```

---

## Reproducing the figures

All main and supplementary figures can be reproduced directly from the pre-computed CSV files in `outputs/` — **no signal data download required**.

```bash
# 1. Install dependencies
pip install numpy scipy pandas matplotlib pingouin pyzaplineplus

# 2. Reproduce main figures from pre-computed features
cd scripts
python 52_bout_figures.py     # Fig 2 — main result (boxplots + heatmap)
python 70_washout.py          # Fig 3 — washout time course
python 90_replot_figures.py   # All figures with unified aesthetics
python 95_s5_stratified.py    # S5 — within-bout fatigue slope
```

To re-run the **full pipeline from preprocessed signals** (requires `.npz` data from Zenodo):

```bash
# After downloading data_derived/ from Zenodo into Paper3/data_derived/
python 22_bout_features.py    # Re-extract per-bout features
python 42_bout_stats.py       # Re-run bout-resolved statistics
python 60_cycle_tf.py         # Cycle-locked TF grand averages
python 62_cycle_tf_cluster.py # Cluster permutation test (Fig 4, S6)
```

To re-run from **raw BDF files** (available on request — see Data Availability):

```bash
pip install mne               # Additional dependency for BDF loading
python 10_preprocess.py       # Raw BDF → data_derived/*.npz
# Then continue from 22_bout_features.py
```

---

## Data availability

| Dataset | Location | Size | Contents |
|---------|----------|------|----------|
| Pre-computed features | `outputs/*.csv` (this repo) | ~180 KB | All features and statistics — sufficient to reproduce all plots |
| Preprocessed EMG signals | [Zenodo — DOI: 10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX) | ~1.4 GB | Per-subject bipolar masseter EMG, 1024 Hz, Zapline-cleaned (`.npz`) |
| Raw recordings (EEG + EMG) | Available on request from corresponding author | ~30 GB | BDF files, 73 channels, 1024 Hz |

> **Why 1024 Hz?** Masticatory EMG spectral analysis requires frequency content up to 450 Hz (median frequency estimation, Morlet TF decomposition). Downsampling would compromise reproducibility of the published results and is explicitly avoided in the pipeline.

---

## Dependencies

```
Python >= 3.10
numpy
scipy
pandas
matplotlib
pingouin          # RM-ANOVA, paired Wilcoxon (Greenhouse-Geisser)
pyzaplineplus     # Zapline-plus line-noise suppression
mne               # BDF file loading (only needed for 10_preprocess.py)
```

Install all at once:

```bash
pip install numpy scipy pandas matplotlib pingouin pyzaplineplus mne
```

---

## Signal processing pipeline

```
Raw BDF (1024 Hz, 73 ch)
    │ 10_preprocess.py
    ▼
Bipolar masseter EMG
 • 20–450 Hz bandpass (zero-phase Butterworth, order 4)
 • Zapline-plus (50/100/150/200 Hz)
 • Bipolar derivation: L = ExG5−ExG6 / R = ExG7−ExG8
 • QC: MDF 55–170 Hz + chewing rate 0.6–2.6 Hz in both conditions
    │ 22_bout_features.py
    ▼
Per-bout features (N=34, 4 bouts × 2 conditions)
 • med_amp, rms_amp, totpow, p_20_60, p_60_150
 • MDF, MNF, chewing rate, CV_IPI, duty cycle
    │ 42_bout_stats.py + 62_cycle_tf_cluster.py
    ▼
Statistics + figures
 • Paired Wilcoxon + BH-FDR per bout
 • RM-ANOVA (condition × bout, Greenhouse-Geisser)
 • Cluster-based permutation test on TF difference maps
```

---

## Ethics

Data collected under ethics approval from the Ethics Committee, School of Dentistry, Universidad de Valparaíso (registration code POSTG-06-22), in accordance with the Declaration of Helsinki. All participants provided written informed consent. This secondary analysis falls within the scope of the original approval.

---

## Citation

If you use this code or data, please cite the paper and the dataset:

```bibtex
@article{espinoza2025emg,
  title   = {Sensorimotor gain compensation during masticatory {EMG} under
             topical dental anaesthesia: a bout-resolved secondary analysis},
  author  = {Espinoza, Sebastian and {Klaus Samson} and Jim\'{e}nez, Sebasti\'{a}n and {El-Deredy}, Wael},
  journal = {[Journal name]},
  year    = {2025},
  doi     = {[DOI upon acceptance]}
}

@dataset{espinoza2025emg_data,
  title   = {Preprocessed masseter {EMG} data — sensorimotor gain compensation study},
  author  = {Espinoza, Sebastian and {Klaus Samson} and Jim\'{e}nez, Sebasti\'{a}n and {El-Deredy}, Wael},
  year    = {2025},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

---

## License

- **Code:** [MIT License](LICENSE)
- **Data:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
