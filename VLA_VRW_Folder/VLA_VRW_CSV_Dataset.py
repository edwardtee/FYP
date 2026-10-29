"""Load VLA_VRW CSV exports into STFT samples.

The intended benchmark protocol is:

    lab  -> development/training data
    real -> held-out robustness/generalizability evaluation data

Each sample is one complete 8-second PERCLOS window.  The loader reads only
the generated CSV files; it does not read EDF or MAT files.
"""

from pathlib import Path
import csv

import cv2
import numpy as np
import pandas as pd
from scipy.signal import spectrogram, butter, sosfiltfilt


CLASS_NAMES = ("alert", "transition", "drowsy")
DEFAULT_CHANNELS = ("Fz_Pz_uv",)


def _read_csv(path, usecols=None):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if usecols is None:
        return rows
    missing = set(usecols) - set(reader.fieldnames or [])
    if missing:
        raise KeyError(f"Missing columns in {path}: {sorted(missing)}")
    return rows


def _stft_image(signal, fs=300.0, nperseg=600, noverlap=300, target_size=(224, 224)):
    """Create one normalized grayscale STFT image from one EEG window."""
    frequencies, _, power = spectrogram(
        signal, fs=fs, nperseg=nperseg, noverlap=noverlap
    )
    mask = (frequencies >= 0.5) & (frequencies <= 30.0)
    power_db = 10.0 * np.log10(power[mask] + 1e-12)
    image = cv2.normalize(power_db, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.resize(image.astype(np.uint8), target_size, interpolation=cv2.INTER_CUBIC)

def butter_bandpass_filter(data, lowcut=0.5, highcut=30.0, fs=300.0, order=4):
    """Numerically stable, zero-phase Butterworth EEG band-pass filter."""
    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError("Cut-offs must satisfy 0 < lowcut < highcut < Nyquist.")
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(data, dtype=np.float64))


def load_csv_recording(eeg_csv, perclos_csv, channels=DEFAULT_CHANNELS):
    """Return STFT images, labels, PERCLOS values and metadata for one recording."""
    perclos_rows = _read_csv(perclos_csv)
    eeg_columns = ["sampling_hz", *channels]
    try:
        eeg_frame = pd.read_csv(eeg_csv, usecols=eeg_columns)
    except ValueError as exc:
        raise KeyError(f"Missing one of {eeg_columns} in {eeg_csv}") from exc
    if eeg_frame.empty or not perclos_rows:
        raise ValueError(f"Empty CSV pair: {eeg_csv}, {perclos_csv}")

    fs = float(eeg_frame["sampling_hz"].iloc[0])
    if not np.isclose(fs, 300.0):
        raise ValueError(f"Expected 300 Hz in {eeg_csv}, found {fs}")
    signals = {
        channel: eeg_frame[channel].to_numpy(dtype=np.float64)
        for channel in channels
    }
    # 1. Apply Butterworth bandpass filter (0.5 - 30 Hz) across the continuous recording
    # to eliminate baseline wander (< 0.5 Hz) and high-frequency muscle/powerline noise (> 30 Hz).
    # 2. Normalize recording-level amplitude (median center, std scale).
    # Remove recording-specific DC offset and amplitude scale before the STFT.
    # This makes the lab-to-real comparison less sensitive to electrode
    # impedance, subject amplitude, and recording-gain differences.
    for channel, signal in signals.items():
        filtered = butter_bandpass_filter(signal, lowcut=0.5, highcut=30.0, fs=fs, order=4)
        center = np.median(filtered)
        scale = np.std(filtered)
        signals[channel] = (filtered - center) / max(scale, 1e-8)

    images, labels, values, metadata = [], [], [], []
    for row in perclos_rows:
        start = int(row["start_sample"])
        end = int(row["end_sample"])
        if end > len(eeg_frame):
            break
        channel_images = [
            _stft_image(signal[start:end], fs=fs)
            for signal in signals.values()
        ]
        # Existing model expects RGB. One channel is replicated; three channels
        # are mapped to RGB in the requested order.
        if len(channel_images) == 1:
            image = np.repeat(channel_images[0][..., None], 3, axis=2)
        elif len(channel_images) == 3:
            image = np.stack(channel_images, axis=2)
        else:
            raise ValueError("Use one channel or exactly three channels for RGB input")
        images.append(image)
        labels.append(int(row["label"]))
        values.append(float(row["perclos"]))
        metadata.append({
            "condition": row["condition"],
            "subject_id": int(row["subject_id"]),
            "window_index": int(row["window_index"]),
        })
    return np.asarray(images), np.asarray(labels), np.asarray(values), metadata


def load_vla_vrw_csv_dataset(csv_root, condition, channels=DEFAULT_CHANNELS):
    """Load all recordings for one condition from the CSV export directory."""
    csv_root = Path(csv_root)
    all_images, all_labels, all_values, all_metadata = [], [], [], []
    eeg_dir = csv_root / condition / "eeg"
    perclos_dir = csv_root / condition / "perclos"
    for eeg_csv in sorted(eeg_dir.glob("*.csv"), key=lambda path: int(path.stem)):
        perclos_csv = perclos_dir / eeg_csv.name
        if not perclos_csv.exists():
            raise FileNotFoundError(f"Missing matching CSV: {perclos_csv}")
        result = load_csv_recording(eeg_csv, perclos_csv, channels=channels)
        images, labels, values, metadata = result
        all_images.append(images)
        all_labels.append(labels)
        all_values.append(values)
        all_metadata.extend(metadata)
    if not all_images:
        raise FileNotFoundError(f"No CSV recordings found under {eeg_dir}")
    return (
        np.concatenate(all_images),
        np.concatenate(all_labels),
        np.concatenate(all_values),
        all_metadata,
    )


def load_benchmark_split(csv_root, channels=DEFAULT_CHANNELS):
    """Return lab training data and real held-out evaluation data."""
    train = load_vla_vrw_csv_dataset(csv_root, "lab", channels=channels)
    test = load_vla_vrw_csv_dataset(csv_root, "real", channels=channels)
    return train, test
