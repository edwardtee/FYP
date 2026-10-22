import mne
import numpy as np
import cv2
from scipy.signal import spectrogram, butter, sosfiltfilt
import os
import glob


def select_model_eeg_channel(channel_names, requested_channel="Fz"):
    """Return the exact EEG channel used by the image-classification model."""
    lookup = {name.casefold(): name for name in channel_names}
    key = requested_channel.casefold()
    if key not in lookup:
        raise ValueError(
            f"Requested EEG channel '{requested_channel}' is unavailable. "
            f"Available channels: {list(channel_names)}"
        )
    return lookup[key]

def extract_stft_spectrogram(eeg_segment, fs=512, nperseg=1024, noverlap=512, target_size=(224, 224)):
    """
    Transforms 1D EEG window into a normalized 2D STFT RGB Spectrogram.
    """
    # At 512 Hz, nperseg=1024 yields 0.5-Hz frequency resolution.  The
    # reported 0.5--4 Hz delta band is therefore physically represented.
    if len(eeg_segment) < nperseg:
        raise ValueError("EEG window is shorter than nperseg.")
    # 1. Compute Short-Time Fourier Transform (STFT)
    frequencies, times, Sxx = spectrogram(eeg_segment, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # 2. Filter frequencies to clinical EEG range (0.5 Hz - 30 Hz: Delta, Theta, Alpha, Beta)
    freq_mask = (frequencies >= 0.5) & (frequencies <= 30.0)
    Sxx = Sxx[freq_mask, :]
    
    # 3. Convert power to log scale (dB)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # 4. Normalize matrix values to standard 8-bit image pixel range [0, 255]
    norm_spec = cv2.normalize(Sxx_db, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_spec = norm_spec.astype(np.uint8)

    # 5. Resize to VGG-16 required spatial dimension (224x224)
    #    Note: scipy.signal.spectrogram returns frequencies in ascending order (row 0 = lowest freq).
    #    Vertical orientation is handled at display time via origin='lower' in all imshow calls.
    resized_spec = cv2.resize(norm_spec, target_size, interpolation=cv2.INTER_CUBIC)

    # 7. Replicate grayscale image across 3 channels to construct RGB format
    rgb_spectrogram = np.stack([resized_spec] * 3, axis=-1)
    
    return rgb_spectrogram

def load_drozy_labels(kss_file_path):
    """
    Parses KSS scores and assigns binary labels.
    KSS <= 5 -> 0 (Alert)
    KSS >= 6 -> 1 (Drowsy)
    """
    kss_scores = np.loadtxt(kss_file_path)
    binary_labels = np.where(kss_scores <= 5, 0, 1)
    return binary_labels

def load_drozy_dataset(dataset_root_path, window_sec=5, overlap_sec=2.5, fs=512,
                       return_groups=False, eeg_channel="Fz", eeg_channels=None):
    """
    Parses DROZY dataset files:
    1. Reads KSS scores from KSS.txt matrix.
    2. Reads EDF signals from psg/ folder using MNE.
    3. Segments signals into sliding window epochs.
    4. Converts each epoch into an STFT image and pairs it with its KSS label.

    ``eeg_channels`` may contain one channel (replicated to RGB for the Fz
    baseline) or exactly three channels (mapped in order to R/G/B for a
    multichannel VGG comparison).
    """
    psg_folder = os.path.join(dataset_root_path, "psg")
    kss_file_path = os.path.join(dataset_root_path, "KSS.txt")
    
    # 1. Load KSS matrix (14 subjects x 3 tests)
    kss_matrix = np.loadtxt(kss_file_path)
    
    spectrograms = []
    labels = []
    groups = []
    if eeg_channels is None:
        eeg_channels = (eeg_channel,)
    eeg_channels = tuple(eeg_channels)
    if len(eeg_channels) not in (1, 3):
        raise ValueError("eeg_channels must contain either one or exactly three channel names.")
    
    # Define window lengths in sample points
    window_samples = int(window_sec * fs)
    step_samples = int((window_sec - overlap_sec) * fs)
    
    # 2. Iterate through all .edf files in psg/
    edf_files = glob.glob(os.path.join(psg_folder, "*.edf"))
    
    for edf_path in edf_files:
        filename = os.path.basename(edf_path) # e.g. "1-1.edf"
        name_parts = filename.replace(".edf", "").split("-")
        
        try:
            subject_id = int(name_parts[0]) # 1 to 14
            test_id = int(name_parts[1])    # 1 to 3
        except ValueError:
            continue  # Skip unformatted files
            
        # Extract corresponding KSS rating (convert 1-based indexing to 0-based)
        kss_score = kss_matrix[subject_id - 1, test_id - 1]
        
        # Skip invalid or missing tests (e.g., test 7-1 has KSS = 0)
        if kss_score <= 0:
            continue
            
        # Binary target labeling: KSS <= 5 -> Alert (0), KSS >= 6 -> Drowsy (1)
        binary_label = 0 if kss_score <= 5 else 1
        
        # 3. Load PSG signal via MNE
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # Channel order is preserved: (Fz, Cz, C3) becomes (R, G, B).
        target_channels = [select_model_eeg_channel(raw.ch_names, name)
                           for name in eeg_channels]
        raw.pick(target_channels)
        recording_fs = float(raw.info["sfreq"])
        if not np.isclose(recording_fs, fs):
            raise ValueError(
                f"{filename} has sampling rate {recording_fs}, expected {fs}. "
                "Resample explicitly before epoching."
            )
        signal_data, _ = raw[:]
        # Filter the entire recording once, before epoching.  Filtering each
        # five-second window separately creates artificial edge transients that
        # can be mistaken for low-frequency EEG activity.
        eeg_signals = butter_bandpass_filter(signal_data, fs=fs)
        
        # 4. Slice signal into sliding windows and convert to STFT spectrograms
        total_samples = eeg_signals.shape[1]
        for start in range(0, total_samples - window_samples + 1, step_samples):
            epochs = eeg_signals[:, start : start + window_samples]
            
            # Fz-only is replicated across RGB.  In the multichannel setting,
            # channel order maps directly to image colour channels.
            channel_specs = [extract_stft_spectrogram(epoch, fs=fs)[..., 0]
                             for epoch in epochs]
            if len(channel_specs) == 1:
                spec_img = np.stack([channel_specs[0]] * 3, axis=-1)
            else:
                spec_img = np.stack(channel_specs, axis=-1)
            
            spectrograms.append(spec_img)
            labels.append(binary_label)
            # Group at participant level so no overlapping windows belonging to
            # one driver can cross between training and held-out evaluation.
            groups.append(subject_id)
            
    result = (np.array(spectrograms), np.array(labels))
    return (*result, np.array(groups)) if return_groups else result

def butter_bandpass_filter(data, lowcut=0.5, highcut=30.0, fs=512, order=4):
    """Numerically stable, zero-phase Butterworth EEG band-pass filter."""
    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError("Cut-offs must satisfy 0 < lowcut < highcut < Nyquist.")
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(data, dtype=np.float64))
