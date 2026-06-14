"""Export raw EMG data in BIDS-EMG format for Zenodo.

Spec: https://bids-specification.readthedocs.io/en/stable/
      modality-specific-files/electromyography.html

Output layout:
  Paper3/zenodo_emg_bids/
    dataset_description.json
    participants.tsv / participants.json
    README
    task-chewing_emg.json         <- inherited sidecar (all subjects)
    task-chewing_channels.tsv     <- inherited channels (all subjects)
    task-chewing_events.json      <- column descriptions
    sub-<label>/
      emg/
        sub-<label>_task-chewing_emg.bdf   (anonymised, 4 EMG ch + Status)
        sub-<label>_task-chewing_events.tsv

Channels exported: EXG5, EXG6, EXG7, EXG8 (raw monopolar, 1024 Hz, V)
Bipolar derivations (Left = EXG5-EXG6, Right = EXG7-EXG8) are computed
in the analysis scripts, NOT applied here.
"""
import csv
import json
import numpy as np
import mne
from pathlib import Path

from config import PROJ, SUBJECTS_36, EMG_LABELS, bout_condition_map
from emg_io import find_subject_file, load_emg

BIDS_DIR = PROJ / "zenodo_emg_bids"

# ── Root metadata ──────────────────────────────────────────────────────────────

DATASET_DESCRIPTION = {
    "Name": "Masseter EMG under topical dental anaesthesia — within-session crossover",
    "BIDSVersion": "1.9.0",
    "License": "CC-BY-4.0",
    "Authors": [
        "Sebastian Espinoza",
        "Wael Almarales-Klaus",
        "Samson Khachatryan",
        "Sebastián Jiménez"
    ],
    "HowToAcknowledge": "Cite the associated paper (DOI pending) and the dataset DOI.",
    "ReferencesAndLinks": [
        "https://github.com/sespinoza-cl/Pap2026-3-emg-ane"
    ],
    "DatasetType": "raw"
}

EMG_SIDECAR = {
    "TaskName": "chewing",
    "TaskDescription": (
        "Within-session crossover (two blocks: anaesthesia and placebo, "
        "counterbalanced). Each block: four 60-second rhythmic gum-chewing bouts "
        "separated by ~4 min cognitive tasks (auditory oddball and verbal 2-back). "
        "Topical lidocaine 10% or matched placebo applied at block onset."
    ),
    "Instructions": (
        "Chew at your comfortable rhythm for 60 seconds when the onset tone sounds."
    ),
    "SamplingFrequency": 1024,
    "PowerLineFrequency": 50,
    "RecordingType": "continuous",
    "EMGPlacementScheme": "ChannelSpecific",
    "EMGReference": "ChannelSpecific",
    "SoftwareFilters": "n/a",
    "HardwareFilters": {
        "BioSemi_ActiveTwo_frontend": {"Description": "DC-coupled, anti-alias filter at Nyquist (512 Hz)"}
    },
    "Manufacturer": "BioSemi",
    "ManufacturersModelName": "ActiveTwo",
    "EMGChannelCount": 4,
    "TriggerChannelCount": 1,
    "ElectrodeMaterial": "Ag/AgCl",
    "ElectrodeType": "cup",
    "SkinPreparation": (
        "Skin cleaned with abrasive paste; electrode gel applied over masseter belly."
    ),
    "InstitutionName": "Universidad de Valparaiso",
    "InstitutionAddress": "Valparaiso, Chile"
}

# Root-level channels.tsv — inherited by all subjects (same placement for all)
CHANNELS_ROWS = [
    {"name": "EXG5", "type": "EMG", "units": "V",
     "description": "Left masseter, positive pole",
     "target_muscle": "masseter", "placement_scheme": "measured",
     "reference": "CMS/DRL", "low_cutoff": "DC", "high_cutoff": "512",
     "status": "good"},
    {"name": "EXG6", "type": "EMG", "units": "V",
     "description": "Left masseter, negative pole",
     "target_muscle": "masseter", "placement_scheme": "measured",
     "reference": "CMS/DRL", "low_cutoff": "DC", "high_cutoff": "512",
     "status": "good"},
    {"name": "EXG7", "type": "EMG", "units": "V",
     "description": "Right masseter, positive pole",
     "target_muscle": "masseter", "placement_scheme": "measured",
     "reference": "CMS/DRL", "low_cutoff": "DC", "high_cutoff": "512",
     "status": "good"},
    {"name": "EXG8", "type": "EMG", "units": "V",
     "description": "Right masseter, negative pole",
     "target_muscle": "masseter", "placement_scheme": "measured",
     "reference": "CMS/DRL", "low_cutoff": "DC", "high_cutoff": "512",
     "status": "good"},
    {"name": "Status", "type": "TRIG", "units": "n/a",
     "description": "BioSemi trigger channel (event codes; see events.tsv)",
     "target_muscle": "n/a", "placement_scheme": "n/a",
     "reference": "n/a", "low_cutoff": "n/a", "high_cutoff": "n/a",
     "status": "good"},
]

CHANNELS_FIELDS = ["name", "type", "units", "description", "target_muscle",
                   "placement_scheme", "reference", "low_cutoff", "high_cutoff",
                   "status"]

# Descriptions for events.tsv columns
EVENTS_JSON = {
    "onset": {"Description": "Event onset relative to recording start", "Units": "s"},
    "duration": {"Description": "Event duration; 60 s for chewing bout onset, 0 for offset/markers", "Units": "s"},
    "trial_type": {
        "Description": "Event category",
        "Levels": {
            "chewing_bout": "Onset of a 60-second chewing bout (event_code=1)",
            "chewing_bout_end": "Offset of a chewing bout (event_code=2)",
            "rest_marker": "Onset of rest/cognitive-task period (event_code=3–6)"
        }
    },
    "event_code": {"Description": "Raw numeric trigger code from the Status channel"},
    "bout_index": {
        "Description": (
            "Sequential bout number within the recording (1–8). "
            "For arm R1: bouts 1-4 = anaesthesia, 5-8 = placebo. "
            "For arm R2: bouts 1-4 = placebo, 5-8 = anaesthesia. "
            "n/a for non-bout events."
        )
    },
    "condition": {
        "Description": "Experimental condition for this bout. n/a for non-bout events.",
        "Levels": {"ANE": "Anaesthesia (lidocaine 10%)", "PLA": "Placebo (saline)"}
    }
}

README_BIDS = """\
Dataset: Masseter EMG under topical dental anaesthesia — within-session crossover
==================================================================================

BIDS version: 1.9.0
Modality: EMG (surface electromyography)
N subjects: 36 (identifiers M1-M7, PS1-PS15, S1-S14)
Final analysis sample: N=34 (M3 and M7 excluded by automated QC; see participants.tsv)

Contents
--------
One BDF file per participant with four raw masseter EMG channels (EXG5–EXG8) at
1024 Hz, plus a Status trigger channel. No filters applied to the exported data.

Channels
--------
  EXG5  Left masseter, positive pole   (monopolar, V, CMS/DRL reference)
  EXG6  Left masseter, negative pole
  EXG7  Right masseter, positive pole
  EXG8  Right masseter, negative pole
  Status  BioSemi trigger channel

Bipolar derivation (Left = EXG5-EXG6, Right = EXG7-EXG8) is computed in the
analysis scripts, not in these files.

Experimental design
-------------------
Within-session crossover: two chewing blocks (anaesthesia and placebo) in
counterbalanced order, separated by ~4 minutes. Each block: four 60-second
chewing bouts (B1-B4) with cognitive tasks (auditory oddball + verbal 2-back)
performed between bouts. See task-chewing_events.json for event code details.

Arm assignment (participants.tsv):
  R1 (n=18): anaesthesia block first (bouts 1-4 = ANE, 5-8 = PLA)
  R2 (n=18): placebo block first     (bouts 1-4 = PLA, 5-8 = ANE)

Recording system
----------------
BioSemi ActiveTwo, 1024 Hz, Ag/AgCl cup electrodes over masseter belly.
Original recording: 64-channel EEG + 8 external electrodes (EXG1-8).
Only masseter channels (EXG5-8) are included in this dataset.

Analysis code & paper
---------------------
https://github.com/sespinoza-cl/Pap2026-3-emg-ane

License
-------
Creative Commons Attribution 4.0 International (CC BY 4.0).
Please cite the associated paper when using these data.
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_tsv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_events_tsv(events, sfreq, rama):
    """Convert raw events array [(sample, code), ...] to BIDS events rows."""
    cond_map = bout_condition_map(rama)   # {0..7: 'ANE'|'PLA'}
    rows = []
    bout_counter = 0  # counts code-1 events seen so far

    for samp, code in events:
        onset = round(samp / sfreq, 4)
        if code == 1:
            condition = cond_map.get(bout_counter, "n/a")
            rows.append({
                "onset": onset,
                "duration": 60.0,
                "trial_type": "chewing_bout",
                "event_code": code,
                "bout_index": bout_counter + 1,
                "condition": condition,
            })
            bout_counter += 1
        elif code == 2:
            rows.append({
                "onset": onset,
                "duration": 0,
                "trial_type": "chewing_bout_end",
                "event_code": code,
                "bout_index": "n/a",
                "condition": "n/a",
            })
        elif 3 <= code <= 6:
            rows.append({
                "onset": onset,
                "duration": 0,
                "trial_type": "rest_marker",
                "event_code": code,
                "bout_index": "n/a",
                "condition": "n/a",
            })
    return rows


EVENTS_FIELDS = ["onset", "duration", "trial_type", "event_code",
                 "bout_index", "condition"]


def export_subject(subj):
    try:
        path = find_subject_file(subj)
    except FileNotFoundError:
        print(f"  {subj}: NOT FOUND — skipping")
        return None

    # BIDS subject directory
    sub_emg_dir = BIDS_DIR / f"sub-{subj}" / "emg"
    sub_emg_dir.mkdir(parents=True, exist_ok=True)

    bdf_out = sub_emg_dir / f"sub-{subj}_task-chewing_emg.bdf"
    ev_out  = sub_emg_dir / f"sub-{subj}_task-chewing_events.tsv"

    # ── Load and pick channels ─────────────────────────────────────────────
    if path.suffix.lower() == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
        status_chs = [ch for ch in raw.ch_names if ch.lower() == "status"]
        picks = mne.pick_channels(raw.ch_names,
                                  include=EMG_LABELS + status_chs)
        raw = raw.pick(picks)
        sfreq = raw.info["sfreq"]
        # Extract events directly from MNE raw before anonymising
        ev_mne = mne.find_events(raw, stim_channel="Status",
                                 verbose="ERROR", shortest_event=1,
                                 consecutive=True)
        ev_mne = ev_mne[ev_mne[:, 2] < 256]
        events_arr = np.column_stack([ev_mne[:, 0], ev_mne[:, 2]]).astype(int)
    else:
        d = load_emg(subj)
        sfreq  = d["sfreq"]
        events_arr = d["events"]
        ch_names = EMG_LABELS + ["Status"]
        ch_types = ["emg"] * 4 + ["stim"]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq,
                               ch_types=ch_types)
        status = np.zeros((1, d["data"].shape[1]), dtype=np.float64)
        for s, c in events_arr:
            if 0 <= s < status.shape[1]:
                status[0, s] = float(c)
        raw = mne.io.RawArray(np.vstack([d["data"], status]), info,
                              verbose="ERROR")

    # ── Determine arm from filename ────────────────────────────────────────
    rama = None
    for tok in path.stem.split("_"):
        if tok in ("R1", "R2"):
            rama = tok[1]  # "1" or "2"
            break
    if rama is None:
        rama = "1"  # fallback

    # ── Anonymise ──────────────────────────────────────────────────────────
    raw.anonymize(daysback=None, keep_his=False, verbose="ERROR")

    # ── Export BDF ─────────────────────────────────────────────────────────
    fmt = "auto"
    out_path = bdf_out
    try:
        raw.export(str(bdf_out), fmt=fmt, overwrite=True, verbose="ERROR")
    except Exception:
        out_path = bdf_out.with_suffix(".edf")
        raw.export(str(out_path), fmt="edf", overwrite=True, verbose="ERROR")

    # ── Write events.tsv ───────────────────────────────────────────────────
    ev_rows = build_events_tsv(events_arr, sfreq, rama)
    write_tsv(ev_out, ev_rows, EVENTS_FIELDS)

    size_mb = round(out_path.stat().st_size / 1e6, 1)
    n_bouts = sum(1 for r in ev_rows if r["trial_type"] == "chewing_bout")
    print(f"  {subj}: {out_path.name}  {size_mb} MB  {n_bouts} bouts  arm=R{rama}")
    return {"subj": subj, "arm": rama, "size_mb": size_mb,
            "n_bouts": n_bouts, "status": "ok"}


def build_participants(results):
    rows = []
    for r in results:
        if r is None:
            continue
        first_cond = "ANE" if r["arm"] == "1" else "PLA"
        rows.append({
            "participant_id": f"sub-{r['subj']}",
            "arm": f"R{r['arm']}",
            "first_condition": first_cond,
            "qc_included": "no" if r["subj"] in ("M3", "M7") else "yes",
        })
    return rows


PARTICIPANTS_FIELDS = ["participant_id", "arm", "first_condition", "qc_included"]

PARTICIPANTS_JSON = {
    "participant_id": {"Description": "Unique participant identifier (sub-<label>)"},
    "arm": {
        "Description": "Counterbalancing arm",
        "Levels": {
            "R1": "Anaesthesia block first (bouts 1-4 = ANE, 5-8 = PLA)",
            "R2": "Placebo block first (bouts 1-4 = PLA, 5-8 = ANE)"
        }
    },
    "first_condition": {
        "Description": "Condition administered in the first block",
        "Levels": {"ANE": "Anaesthesia (lidocaine 10%)", "PLA": "Placebo (saline)"}
    },
    "qc_included": {
        "Description": "Included in main analyses after automated EMG QC",
        "Levels": {"yes": "Included (N=34)", "no": "Excluded: dead electrode in one condition"}
    }
}


def main():
    BIDS_DIR.mkdir(exist_ok=True)
    print(f"BIDS output: {BIDS_DIR}")
    print(f"Exporting {len(SUBJECTS_36)} subjects...\n")

    # ── Root files ─────────────────────────────────────────────────────────
    write_json(BIDS_DIR / "dataset_description.json", DATASET_DESCRIPTION)
    write_json(BIDS_DIR / "task-chewing_emg.json", EMG_SIDECAR)
    write_json(BIDS_DIR / "task-chewing_events.json", EVENTS_JSON)
    (BIDS_DIR / "README").write_text(README_BIDS, encoding="utf-8")
    write_tsv(BIDS_DIR / "task-chewing_channels.tsv", CHANNELS_ROWS,
              CHANNELS_FIELDS)

    # ── Subjects ───────────────────────────────────────────────────────────
    results = []
    for subj in SUBJECTS_36:
        print(f"Processing {subj}...")
        results.append(export_subject(subj))

    # ── Participants TSV/JSON ──────────────────────────────────────────────
    p_rows = build_participants(results)
    write_tsv(BIDS_DIR / "participants.tsv", p_rows, PARTICIPANTS_FIELDS)
    write_json(BIDS_DIR / "participants.json", PARTICIPANTS_JSON)

    # ── Summary ────────────────────────────────────────────────────────────
    ok = sum(1 for r in results if r is not None)
    total_mb = sum(r["size_mb"] for r in results if r is not None)
    print(f"\nDone: {ok}/{len(SUBJECTS_36)} subjects exported")
    print(f"Total EMG data: {total_mb:.0f} MB ({total_mb/1024:.2f} GB)")
    print(f"Output: {BIDS_DIR}")
    print("\nValidate with: bids-validator zenodo_emg_bids/")


if __name__ == "__main__":
    main()
