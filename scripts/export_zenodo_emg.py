"""Export raw EMG-only BDF files for Zenodo data sharing.

Extracts channels EXG5--EXG8 (bilateral masseter) + Status (events) from
each participant's raw recording, anonymises the BDF header, and writes a
minimal BDF file to zenodo_emg/.

Input  : D:/Exp1/.../All_36/<subj>_*.bdf  (or .set)
Output : Paper3/zenodo_emg/<subj>_emg.bdf   (4 EMG ch + Status, 1024 Hz)
         Paper3/zenodo_emg/README.txt
         Paper3/zenodo_emg/MANIFEST.csv

Channel mapping preserved in output:
  EXG5 = Left masseter (+)
  EXG6 = Left masseter (-)
  EXG7 = Right masseter (+)
  EXG8 = Right masseter (-)
Bipolar derivations: Left = EXG5-EXG6, Right = EXG7-EXG8
(computed in downstream analysis scripts, not applied here)

Event codes relevant to this study:
  1 = Chewing bout ON
  2 = Chewing bout OFF
  3-6 = Rest/task markers
"""
import csv
import os
import numpy as np
import mne
from pathlib import Path

from config import PROJ, SUBJECTS_36, EMG_LABELS, FS_ORIG
from emg_io import find_subject_file

OUT_DIR = PROJ / "zenodo_emg"
OUT_DIR.mkdir(exist_ok=True)

README = """\
Dataset: Masseter EMG under topical dental anesthesia — Paper 3
===============================================================

Contents
--------
One BDF file per participant (N=36; identifiers M1-M7, PS1-PS15, S1-S14).
Each file contains four channels of raw surface EMG from bilateral masseter
muscles at 1024 Hz, plus a Status channel encoding task triggers.

Channels
--------
  EXG5  Left masseter electrode (+pole)
  EXG6  Left masseter electrode (-pole)
  EXG7  Right masseter electrode (+pole)
  EXG8  Right masseter electrode (-pole)
  Status  BioSemi trigger channel (event codes below)

Bipolar derivation (Left = EXG5-EXG6, Right = EXG7-EXG8) is applied in
the analysis scripts, not in these files.

Event codes
-----------
  1   Chewing bout onset  (60-s duration; 4 bouts per condition block)
  2   Chewing bout offset
  3-6 Rest / cognitive-task period markers

Experimental design
-------------------
Within-session crossover: participants completed two chewing blocks
(Anesthesia and Placebo) in counterbalanced order, separated by ~4 min.
Block identity (ANE/PLA) is encoded in the filename: R1 files contain
Anesthesia block first (bouts 1-4 = ANE, 5-8 = PLA); R2 files the reverse.
See MANIFEST.csv for per-participant arm assignment.

Recording system
----------------
BioSemi ActiveTwo, 1024 Hz, Ag/AgCl electrodes over masseter belly.
Full 64-channel EEG+EMG recording; only masseter channels exported here.

Analysis code
-------------
https://github.com/sespinoza-cl/Pap2026-3-emg-ane

License
-------
Creative Commons Attribution 4.0 International (CC BY 4.0).
Please cite the associated paper when using these data.
"""

MANIFEST_FIELDS = ["subj", "file", "rama", "n_samples", "duration_min",
                   "n_events_code1", "size_mb", "status"]


def export_subject(subj):
    try:
        path = find_subject_file(subj)
    except FileNotFoundError:
        print(f"  {subj}: NOT FOUND — skipping")
        return {"subj": subj, "file": "", "rama": "", "n_samples": "",
                "duration_min": "", "n_events_code1": "", "size_mb": "",
                "status": "missing"}

    rama = None
    for tok in path.stem.split("_"):
        if tok in ("R1", "R2"):
            rama = tok
            break

    out_path = OUT_DIR / f"{subj}_emg.bdf"

    if path.suffix.lower() == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")

        # Pick EMG channels + Status
        emg_picks = [raw.ch_names.index(l) for l in EMG_LABELS]
        status_picks = [i for i, ch in enumerate(raw.ch_names)
                        if ch.lower() in ("status", "stim")]
        picks = mne.pick_channels(
            raw.ch_names,
            include=EMG_LABELS + [raw.ch_names[i] for i in status_picks],
        )
        raw = raw.pick(picks)

    else:
        # SET file — reconstruct minimal Raw from numpy via emg_io
        from emg_io import load_emg
        d = load_emg(subj)
        fs = d["sfreq"]
        emg = d["data"]          # (4, n) in V
        events = d["events"]     # (k, 2) = [sample, code]

        ch_names = EMG_LABELS + ["Status"]
        ch_types = ["emg"] * 4 + ["stim"]
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
        info.set_montage(None)

        # Build status channel from events
        status = np.zeros((1, emg.shape[1]), dtype=np.float64)
        for samp, code in events:
            if 0 <= samp < status.shape[1]:
                status[0, samp] = float(code)

        data = np.vstack([emg, status])
        raw = mne.io.RawArray(data, info, verbose="ERROR")

    # Anonymise: wipe personal fields from header
    raw.anonymize(daysback=None, keep_his=False, verbose="ERROR")

    # Export to BDF
    try:
        raw.export(str(out_path), fmt="auto", overwrite=True, verbose="ERROR")
    except Exception:
        # Fallback: save as EDF if BDF export not available in this MNE version
        out_path = out_path.with_suffix(".edf")
        raw.export(str(out_path), fmt="edf", overwrite=True, verbose="ERROR")

    # Collect manifest info
    ev = mne.find_events(raw, stim_channel="Status", verbose="ERROR",
                         shortest_event=1, consecutive=True) if "Status" in raw.ch_names else np.empty((0, 3))
    n_code1 = int(np.sum(ev[:, 2] == 1)) if ev.size else 0
    n_samp = raw.n_times
    dur = round(n_samp / raw.info["sfreq"] / 60, 2)
    size_mb = round(out_path.stat().st_size / 1e6, 1)

    print(f"  {subj}: {out_path.name}  {dur} min  {n_code1} bouts  {size_mb} MB")
    return {"subj": subj, "file": out_path.name, "rama": rama or "?",
            "n_samples": n_samp, "duration_min": dur,
            "n_events_code1": n_code1, "size_mb": size_mb, "status": "ok"}


def main():
    print(f"Exporting {len(SUBJECTS_36)} subjects → {OUT_DIR}")

    rows = []
    for subj in SUBJECTS_36:
        print(f"Processing {subj}...")
        rows.append(export_subject(subj))

    # Write README
    (OUT_DIR / "README.txt").write_text(README, encoding="utf-8")

    # Write MANIFEST
    manifest_path = OUT_DIR / "MANIFEST.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    total_mb = sum(r["size_mb"] for r in rows if r["status"] == "ok")
    print(f"\nDone: {ok}/{len(SUBJECTS_36)} subjects exported")
    print(f"Total size: {total_mb:.0f} MB ({total_mb/1024:.2f} GB)")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
