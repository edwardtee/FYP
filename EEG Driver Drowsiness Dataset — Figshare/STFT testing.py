import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# 1. Load the .mat file
mat_file_path = 'dataset.mat'  # Replace with your actual file path
data = sio.loadmat(mat_file_path)

# 2. Extract the variables
# scipy.io adds metadata keys like '__header__', '__version__', '__globals__'
eeg_samples = data['EEGsample']  # EEG signal tensor
labels = data['substate']        # Labels: 0 = Alert, 1 = Drowsy
subject_ids = data['subindex']   # Subject IDs (1 to 11)

# Flatten labels and subject arrays for easier handling
labels = labels.ravel()
subject_ids = subject_ids.ravel()

# 3. Print shapes and inspect the dataset structure
print("=== Dataset Summary ===")
print(f"EEG Samples Shape:  {eeg_samples.shape}")  # Expecting (2022, 30, 384) or similar
print(f"Labels Shape:       {labels.shape}")       # Expecting (2022,)
print(f"Subject IDs Shape:  {subject_ids.shape}")  # Expecting (2022,)

print("\n=== Class Distribution ===")
print(f"Alert Samples (0):  {np.sum(labels == 0)}")
print(f"Drowsy Samples (1): {np.sum(labels == 1)}")

print("\n=== Unique Subjects ===")
print(f"Subject IDs present: {np.unique(subject_ids)}")

# Select the first sample and Channel 0 (e.g., Fp1 or Fz)
sample_idx = 0
channel_idx = 0
eeg_signal = eeg_samples[sample_idx, channel_idx, :]  # 384 time points

# Parameters
sampling_rate = 128  # 128 Hz

# Compute Short-Time Fourier Transform (STFT)
frequencies, times, Sxx = spectrogram(eeg_signal, fs=sampling_rate, nperseg=64, noverlap=32)

# Plot and save as 2D Spectrogram Image
plt.figure(figsize=(6, 4))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title(f'Sample {sample_idx} (Label: {"Drowsy" if labels[sample_idx] == 1 else "Alert"})')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.tight_layout()
plt.savefig('eeg_spectrogram_example.png', dpi=300)
plt.show()