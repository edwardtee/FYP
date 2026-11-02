"""Visualize raw and filtered EEG signals directly from an EDF file."""

import concurrent.futures  # Eagerly import to prevent Python 3.13 atexit shutdown race condition
from pathlib import Path
import filelock  # Eagerly import for MNE configuration locks
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt


def butter_bandpass_filter(data, lowcut=0.5, highcut=30.0, fs=300.0, order=4):
    """Zero-phase Butterworth bandpass filter."""
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(data, dtype=np.float64))


def plot_eeg_window(
    edf_path="D:/FYP/VLA_VRW/lab/EEG/1.edf",
    channel="EEG Fz - Pz",
    start_sec=32.0,
    duration_sec=8.0,
    save_path="D:/FYP/VLA_VRW_Folder/eeg_lab1_sample.png",
):
    """Extract a window from the EDF file and plot Raw vs Filtered EEG."""
    print(f"Loading {edf_path}...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    fs = float(raw.info["sfreq"])
    if channel not in raw.ch_names:
        raise ValueError(f"Channel '{channel}' not found. Available: {raw.ch_names}")

    # Convert Volts (MNE default) to microvolts (uV)
    start_sample = int(start_sec * fs)
    end_sample = int((start_sec + duration_sec) * fs)
    raw_signal_uV = raw.get_data(picks=channel, start=start_sample, stop=end_sample)[0] * 1e6

    # Apply 0.5-30 Hz Butterworth filter
    filtered_signal_uV = butter_bandpass_filter(raw_signal_uV, lowcut=0.5, highcut=30.0, fs=fs, order=4)

    # Time vector for plotting
    time = np.linspace(start_sec, start_sec + duration_sec, len(raw_signal_uV))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 1. Raw EEG
    axes[0].plot(time, raw_signal_uV, color="#1f77b4", linewidth=1.0)
    axes[0].set_title(f"Lab Subject 1: Raw EEG Signal ({channel})", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amplitude (uV)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Filtered EEG
    axes[1].plot(time, filtered_signal_uV, color="#2ca02c", linewidth=1.0)
    axes[1].set_title(f"Lab Subject 1: Filtered EEG (0.5–30 Hz Butterworth, {channel})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Time (seconds)", fontsize=10)
    axes[1].set_ylabel("Amplitude (uV)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved successfully to: {Path(save_path).resolve()}")
    plt.close(fig)


def interactive_browser(edf_path="D:/FYP/VLA_VRW/lab/EEG/1.edf"):
    """Launch MNE's interactive scrollable signal browser."""
    print(f"Loading {edf_path} for interactive viewing...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Pick only the EEG channels (exclude Trigger/Annotations)
    eeg_channels = [ch for ch in raw.ch_names if ch.startswith("EEG")]
    raw.pick(eeg_channels)

    # block=True prevents Python from shutting down while the window is still open
    raw.plot(
        duration=10.0,
        n_channels=8,
        scalings="auto",
        title="Lab Subject 1 EEG Browser",
        block=True,
    )


if __name__ == "__main__":
    # Choose what you want to do:
    # 1. To generate and save the static comparison PNG image:
    '''plot_eeg_window(
        edf_path="D:/FYP/VLA_VRW/lab/EEG/1.edf",
        channel="EEG Fz - Pz",
        start_sec=32.0,
        duration_sec=8.0,
        save_path="D:/FYP/VLA_VRW_Folder/eeg_lab1_sample.png",
    )'''

    # 2. To open the interactive scrollable browser, uncomment the line below:
    interactive_browser(edf_path="D:/FYP/VLA_VRW/lab/EEG/1.edf")
