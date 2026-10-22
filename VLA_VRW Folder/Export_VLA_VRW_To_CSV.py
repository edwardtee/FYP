"""Export VLA_VRW EDF and PERCLOS recordings to aligned CSV files.

The exporter writes two files per recording:

    <output>/<condition>/eeg/<subject>.csv
    <output>/<condition>/perclos/<subject>.csv

EEG values are written in microvolts.  The first 24 seconds are removed by
default, as recommended by VLA_VRW/README.txt.  The corresponding first three
8-second PERCLOS windows are removed as well.  The remaining EEG and PERCLOS
timelines are truncated to their common duration.  PERCLOS is kept at
its native one-value-per-window resolution; it is not incorrectly repeated as
if it were a 300 Hz signal.
"""

from argparse import ArgumentParser
from pathlib import Path
import csv
import re

import mne
import numpy as np
from scipy.io import loadmat


PERCLOS_WINDOW_SEC = 8.0
PERCLOS_CLASSES = (
    (0.35, "alert", 0),       # perclos < 0.35
    (0.70, "transition", 1),  # 0.35 <= perclos <= 0.70
    (np.inf, "drowsy", 2),    # perclos > 0.70
)


def perclos_class(value):
    """Return (numeric_label, label_name) using the requested boundaries."""
    value = float(value)
    if value < 0.35:
        return 0, "alert"
    if value <= 0.70:
        return 1, "transition"
    return 2, "drowsy"


def _subject_id(edf_path):
    match = re.fullmatch(r"(\d+)\.edf", edf_path.name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unexpected EDF filename: {edf_path.name}")
    return int(match.group(1))


def _eeg_channel_names(raw):
    """Select EEG channels and create stable CSV column names."""
    selected = []
    csv_names = []
    for channel in raw.ch_names:
        if not channel.startswith("EEG "):
            continue
        # Example: EEG Fz - Pz -> Fz_Pz
        simplified = re.sub(r"^EEG\s+", "", channel).strip()
        simplified = re.sub(r"\s*-\s*", "_", simplified)
        simplified = re.sub(r"\s+", "_", simplified)
        selected.append(channel)
        csv_names.append(simplified)
    if not selected:
        raise ValueError(f"No EEG channels found in {raw.filenames[0]}")
    return selected, csv_names


def _read_perclos(mat_path):
    mat = loadmat(mat_path)
    if "perclos" not in mat:
        raise KeyError(f"{mat_path} does not contain a 'perclos' variable")
    values = np.asarray(mat["perclos"], dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{mat_path} contains an empty PERCLOS array")
    return values


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def export_recording(condition_dir, output_dir, edf_path, discard_initial_sec=24.0):
    subject_id = _subject_id(edf_path)
    mat_path = condition_dir / "perclos" / f"{subject_id}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing matching PERCLOS file: {mat_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    eeg_channels, csv_channels = _eeg_channel_names(raw)
    raw.pick(eeg_channels)
    fs = float(raw.info["sfreq"])
    if not np.isclose(fs, 300.0):
        raise ValueError(f"Expected 300 Hz in {edf_path}, found {fs}")

    perclos = _read_perclos(mat_path)
    discard_samples = int(round(discard_initial_sec * fs))
    discard_perclos_windows = int(round(discard_initial_sec / PERCLOS_WINDOW_SEC))
    if not np.isclose(
        discard_perclos_windows * PERCLOS_WINDOW_SEC,
        discard_initial_sec,
    ):
        raise ValueError(
            "discard_initial_sec must be an exact multiple of the "
            f"{PERCLOS_WINDOW_SEC:g}-second PERCLOS window"
        )
    if discard_perclos_windows >= perclos.size:
        raise ValueError(f"Discard period removes all PERCLOS data: {mat_path}")
    perclos = perclos[discard_perclos_windows:]
    eeg_uv = raw.get_data() * 1e6
    if discard_samples >= eeg_uv.shape[1]:
        raise ValueError(f"Discard period removes the complete recording: {edf_path}")
    eeg_uv = eeg_uv[:, discard_samples:]

    # PERCLOS is one value for each 8-second interval.  Retain only complete
    # intervals that also have EEG samples, following the README alignment rule.
    complete_perclos = min(
        perclos.size,
        int(eeg_uv.shape[1] // round(PERCLOS_WINDOW_SEC * fs)),
    )
    if complete_perclos == 0:
        raise ValueError(f"No aligned 8-second windows remain for {edf_path}")
    aligned_samples = int(complete_perclos * PERCLOS_WINDOW_SEC * fs)
    eeg_uv = eeg_uv[:, :aligned_samples]
    perclos = perclos[:complete_perclos]

    condition = condition_dir.name
    eeg_header = [
        "sample_index", "time_s", "sampling_hz", "subject_id", "condition",
        *[f"{name}_uv" for name in csv_channels],
    ]
    eeg_rows = []
    for sample_index in range(aligned_samples):
        eeg_rows.append([
            sample_index,
            sample_index / fs,
            fs,
            subject_id,
            condition,
            *eeg_uv[:, sample_index].tolist(),
        ])

    perclos_header = [
        "window_index", "start_s", "end_s", "start_sample", "end_sample",
        "perclos", "label", "label_name", "sampling_hz", "subject_id", "condition",
    ]
    perclos_rows = []
    for window_index, value in enumerate(perclos):
        label, label_name = perclos_class(value)
        start_sample = int(window_index * PERCLOS_WINDOW_SEC * fs)
        end_sample = int((window_index + 1) * PERCLOS_WINDOW_SEC * fs)
        perclos_rows.append([
            window_index,
            window_index * PERCLOS_WINDOW_SEC,
            (window_index + 1) * PERCLOS_WINDOW_SEC,
            start_sample,
            end_sample,
            float(value),
            label,
            label_name,
            fs,
            subject_id,
            condition,
        ])

    eeg_path = output_dir / condition / "eeg" / f"{subject_id}.csv"
    perclos_path = output_dir / condition / "perclos" / f"{subject_id}.csv"
    _write_csv(eeg_path, eeg_header, eeg_rows)
    _write_csv(perclos_path, perclos_header, perclos_rows)
    print(
        f"{condition}/{subject_id}: {len(csv_channels)} EEG channels, "
        f"{aligned_samples} samples, {complete_perclos} PERCLOS windows"
    )


def export_dataset(dataset_root, output_dir, discard_initial_sec=24.0):
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    for condition in ("lab", "real"):
        condition_dir = dataset_root / condition
        for edf_path in sorted(
            (condition_dir / "EEG").glob("*.edf"),
            key=lambda path: int(path.stem),
        ):
            export_recording(
                condition_dir,
                output_dir,
                edf_path,
                discard_initial_sec=discard_initial_sec,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Export VLA_VRW EDF/PERCLOS data to aligned CSV files.")
    parser.add_argument("--dataset-root", default=r"D:\FYP\VLA_VRW")
    parser.add_argument("--output-dir", default=r"D:\FYP\VLA_VRW_CSV")
    parser.add_argument("--discard-initial-sec", type=float, default=24.0)
    args = parser.parse_args()
    export_dataset(args.dataset_root, args.output_dir, args.discard_initial_sec)
