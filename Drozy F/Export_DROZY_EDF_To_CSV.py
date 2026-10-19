"""Export DROZY EDF recordings to labelled CSV files.

The classifier currently uses Fz only.  This script exports Fz by default,
with both raw and 0.5--30 Hz zero-phase filtered signal in microvolts.  Use
--all-eeg to export all available scalp EEG channels for future multichannel
experiments.  One CSV is created per subject-session recording, retaining
metadata and the recording-level KSS-derived binary label on every row.
"""

from argparse import ArgumentParser
from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd

from STFT_Spectrogram_Generation import butter_bandpass_filter, select_model_eeg_channel


SCALP_EEG_CHANNELS = ("Fz", "Cz", "C3", "C4", "Pz", "Oz")


def _recording_metadata(edf_path, kss_matrix):
    match = re.fullmatch(r"(\d+)-(\d+)\.edf", edf_path.name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unexpected EDF filename: {edf_path.name}")
    subject_id, session_id = map(int, match.groups())
    kss = float(kss_matrix[subject_id - 1, session_id - 1])
    if kss <= 0:
        return subject_id, session_id, kss, None
    return subject_id, session_id, kss, int(kss >= 6)


def export_drozy_to_csv(dataset_root, output_dir, all_eeg=False, model_channel="Fz"):
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kss_matrix = np.loadtxt(dataset_root / "KSS.txt")

    for edf_path in sorted((dataset_root / "psg").glob("*.edf")):
        subject_id, session_id, kss, label = _recording_metadata(edf_path, kss_matrix)
        if label is None:
            print(f"Skipping {edf_path.name}: invalid/missing KSS ({kss:g})")
            continue

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        if all_eeg:
            selected = [ch for ch in SCALP_EEG_CHANNELS if ch in raw.ch_names]
            if not selected:
                raise ValueError(f"No expected scalp EEG channel in {edf_path.name}")
        else:
            selected = [select_model_eeg_channel(raw.ch_names, model_channel)]

        raw.pick(selected)
        fs = float(raw.info["sfreq"])
        signals_uv = raw.get_data() * 1e6  # MNE EDF data are represented in volts.
        frame = {
            "time_s": raw.times,
            "subject_id": np.full(raw.n_times, subject_id, dtype=np.int16),
            "session_id": np.full(raw.n_times, session_id, dtype=np.int8),
            "sampling_hz": np.full(raw.n_times, fs),
            "kss_score": np.full(raw.n_times, kss),
            "label": np.full(raw.n_times, label, dtype=np.int8),
            "label_name": np.full(raw.n_times, "Drowsy" if label else "Alert"),
        }
        for channel, signal_uv in zip(selected, signals_uv):
            frame[f"{channel}_raw_uv"] = signal_uv
            frame[f"{channel}_filtered_0.5_30hz_uv"] = (
                butter_bandpass_filter(signal_uv, fs=fs)
            )

        csv_path = output_dir / f"subject_{subject_id:02d}_session_{session_id}.csv"
        pd.DataFrame(frame).to_csv(csv_path, index=False)
        print(f"Exported {csv_path.name}: {selected}, KSS={kss:g}, label={label}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Export labelled DROZY EDF EEG to CSV.")
    parser.add_argument("--dataset-root", default=r"D:\FYP\DROZY\DROZY")
    parser.add_argument("--output-dir", default=r"D:\FYP\DROZY\DROZY_EEG_CSV")
    parser.add_argument("--all-eeg", action="store_true",
                        help="Export Fz, Cz, C3, C4, Pz and Oz when available.")
    parser.add_argument("--model-channel", default="Fz",
                        help="Single channel to export by default (default: Fz).")
    args = parser.parse_args()
    export_drozy_to_csv(args.dataset_root, args.output_dir,
                        all_eeg=args.all_eeg, model_channel=args.model_channel)
