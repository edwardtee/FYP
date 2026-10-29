"""Grad-CAM for the CNN part of the CNN-LSTM EEG model.

This module is intentionally separate from ``Grad_CAM_Explainability.py``.
It produces one heatmap for each spectrogram in a temporal sequence and
explains one ordinal threshold at a time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalise_heatmap(cam):
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)


def generate_cnn_lstm_gradcam(model, sequence, target_threshold=1,
                              target_layer=None, device="cuda"):
    """Generate a separate CNN Grad-CAM heatmap for every sequence frame.

    Parameters
    ----------
    sequence : torch.Tensor
        One sequence with shape ``(T, C, H, W)`` or ``(1, T, C, H, W)``.
    target_threshold : int
        Ordinal output to explain: 0 means ``class >= Transition`` and 1
        means ``class >= Drowsy``.
    target_layer : torch.nn.Module, optional
        CNN convolutional layer. The final CNN convolution is used by default.

    Returns
    -------
    heatmaps : numpy.ndarray
        Array with shape ``(T, H, W)``.
    predicted_class : int
        Predicted ordinal class using thresholds ``(0.5, 0.5)``.
    cumulative_probabilities : numpy.ndarray
        The two ordinal probabilities.
    """
    device = torch.device(
        device if torch.cuda.is_available() or str(device) == "cpu" else "cpu"
    )
    model = model.to(device)
    was_training = model.training
    model.eval()

    if sequence.ndim == 4:
        sequence = sequence.unsqueeze(0)
    if sequence.ndim != 5 or sequence.shape[0] != 1:
        raise ValueError("sequence must have shape [T,C,H,W] or [1,T,C,H,W]")
    if target_threshold not in (0, 1):
        raise ValueError("target_threshold must be 0 or 1")

    sequence = sequence.to(device).float()
    sequence.requires_grad_(True)
    target_layer = target_layer or next(
        layer for layer in reversed(model.features)
        if isinstance(layer, nn.Conv2d)
    )

    activations = []
    gradients = []

    def save_activations(_module, _inputs, output):
        activations.append(output)

    def save_gradients(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    relu_layers = [layer for layer in model.modules() if isinstance(layer, nn.ReLU)]
    original_inplace = [layer.inplace for layer in relu_layers]
    for layer in relu_layers:
        layer.inplace = False

    forward_handle = target_layer.register_forward_hook(save_activations)
    backward_handle = target_layer.register_full_backward_hook(save_gradients)

    try:
        model.zero_grad(set_to_none=True)
        # Disable cuDNN so the LSTM backward pass works in eval mode.
        with torch.backends.cudnn.flags(enabled=False):
            logits = model(sequence)
            cumulative = torch.sigmoid(logits)
            predicted_class = int((cumulative >= 0.5).sum(dim=1).item())
            logits[0, target_threshold].backward()

        # The CNN receives T frames as a flattened batch. Restore the time
        # dimension before calculating a separate CAM for every frame.
        activation = activations[0]
        gradient = gradients[0]
        time_steps = sequence.shape[1]
        if activation.shape[0] != time_steps:
            raise RuntimeError(
                "Unexpected CNN activation batch size; this function expects "
                "a batch size of one sequence."
            )
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cams = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cams = F.interpolate(
            cams, size=sequence.shape[-2:], mode="bilinear", align_corners=False
        )
        heatmaps = np.stack([
            _normalise_heatmap(cams[time_index, 0].detach().cpu().numpy())
            for time_index in range(time_steps)
        ])
        return (
            heatmaps,
            predicted_class,
            cumulative[0].detach().cpu().numpy(),
        )
    finally:
        forward_handle.remove()
        backward_handle.remove()
        sequence.requires_grad_(False)
        for layer, inplace in zip(relu_layers, original_inplace):
            layer.inplace = inplace
        if was_training:
            model.train()


def show_cnn_lstm_gradcam_example(model, data_loader, device="cuda",
                                  target_threshold=1, sequence_index=0,
                                  save_path="gradcam_cnn_lstm.png"):
    """Save a five-frame Grad-CAM figure for one CNN-LSTM sequence."""
    sequences, labels = next(iter(data_loader))
    if sequence_index >= len(sequences):
        raise IndexError(
            f"sequence_index must be between 0 and {len(sequences) - 1}"
        )

    sequence = sequences[sequence_index]
    true_class = int(labels[sequence_index].item())
    heatmaps, predicted_class, cumulative = generate_cnn_lstm_gradcam(
        model,
        sequence,
        target_threshold=target_threshold,
        device=device,
    )

    frame_count = sequence.shape[0]
    fig, axes = plt.subplots(frame_count, 3, figsize=(12, 3.2 * frame_count))
    axes = np.atleast_2d(axes)
    for time_index in range(frame_count):
        image = sequence[time_index].detach().cpu().numpy().transpose(1, 2, 0).mean(axis=2)
        image = _normalise_heatmap(image)
        axes[time_index, 0].imshow(image, cmap="viridis", origin="lower", aspect="auto")
        axes[time_index, 0].set_title(f"Frame {time_index + 1}: input")
        axes[time_index, 1].imshow(heatmaps[time_index], cmap="jet", origin="lower", aspect="auto")
        axes[time_index, 1].set_title("Grad-CAM")
        axes[time_index, 2].imshow(image, cmap="gray", origin="lower", aspect="auto")
        axes[time_index, 2].imshow(
            heatmaps[time_index], cmap="jet", alpha=0.45,
            origin="lower", aspect="auto"
        )
        axes[time_index, 2].set_title("Overlay")

        # 2. Map frequency values to row indices
        num_freq_bins = image.shape[0]  # total height of image matrix
        freq_bounds = [0.5, 4.0, 8.0, 13.0, 30.0]

        # Calculate pixel indices corresponding to each frequency boundary
        # (assuming 0 corresponds to 0.5 Hz and num_freq_bins-1 corresponds to 30.0 Hz)
        y_indices = [(f - 0.5) / (30.0 - 0.5) * (num_freq_bins - 1) for f in freq_bounds]
        for axis in axes[time_index]:
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Frequency (Hz)")
            axis.set_yticks(y_indices)
            axis.set_yticklabels([f"{f} Hz" for f in freq_bounds])

    target_name = "class >= Drowsy" if target_threshold == 1 else "class >= Transition"
    fig.suptitle(
        f"CNN-LSTM Grad-CAM: explaining {target_name}\n"
        f"True centre class: {true_class}; predicted class: {predicted_class}; "
        f"ordinal probabilities: {np.round(cumulative, 3)}"
    )
    return heatmaps, predicted_class, cumulative


# ---------------------------------------------------------------------------
# Quantitative Grad-CAM Quality Metrics (CNN-LSTM)
# ---------------------------------------------------------------------------

FREQ_MIN_HZ, FREQ_MAX_HZ, IMAGE_HEIGHT = 0.5, 30.0, 224

def _freq_to_row(freq_hz, height=IMAGE_HEIGHT):
    """Convert Hz to row index in the spectrogram (row 0 = 0.5 Hz bottom)."""
    frac = (freq_hz - FREQ_MIN_HZ) / (FREQ_MAX_HZ - FREQ_MIN_HZ)
    return int(round(frac * height))

def get_band_rows(height=IMAGE_HEIGHT):
    """Returns row slice ranges for standard EEG frequency bands."""
    return {
        "Delta (0.5-4 Hz)":  (0,                         _freq_to_row(4, height)),
        "Theta (4-8 Hz)":    (_freq_to_row(4, height),   _freq_to_row(8, height)),
        "Alpha (8-13 Hz)":   (_freq_to_row(8, height),   _freq_to_row(13, height)),
        "Beta  (13-30 Hz)":  (_freq_to_row(13, height),  height),
    }

BAND_COLORS = {
    "Delta (0.5-4 Hz)":  "#4e79a7",
    "Theta (4-8 Hz)":    "#f28e2b",
    "Alpha (8-13 Hz)":   "#59a14f",
    "Beta  (13-30 Hz)":  "#e15759",
}

def _gini(values):
    """Gini coefficient of a flat array (higher -> sparser / more localized)."""
    v = np.abs(values.flatten())
    v = np.sort(v)
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)

def _cosine_sim(a, b):
    """Cosine similarity between two attribution heatmaps."""
    a, b = a.flatten(), b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def _plot_cnn_lstm_gradcam_quantitative_summary(results, gini_vals, cos_sims,
                                                auc_orig, auc_occ, N,
                                                title="Quantitative Grad-CAM Evaluation (CNN-LSTM)",
                                                save_path="cnn_lstm_gradcam_quantitative_evaluation.png"):
    """Produces a 4-panel standalone summary figure for quantitative CNN-LSTM Grad-CAM metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), gridspec_kw={"wspace": 0.45})
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    # --- (A) Fidelity bar ---
    ax = axes[0]
    vals_a = [auc_orig, auc_occ]
    colors_a = ["#4e79a7", "#f28e2b"]
    bars = ax.bar(["Original AUC", "Occluded AUC"], vals_a,
                  color=colors_a, edgecolor="black", linewidth=0.7, width=0.5)
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    drop_val = results.get("fidelity_auc_drop", float("nan"))
    drop_str = f"{drop_val:+.4f}" if not np.isnan(drop_val) else "N/A"
    ax.set_title(f"(A) Fidelity\n(AUC drop = {drop_str})", fontsize=10, fontweight="bold")
    ax.set_ylabel("AUC / Prediction Probability")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8)
    drop_color = "darkgreen" if (not np.isnan(drop_val) and drop_val > 0) else "darkred"
    ax.text(0.5, 0.03, f"Delta = {drop_str}",
            ha="center", transform=ax.transAxes, fontsize=9, color=drop_color, fontweight="bold")

    # --- (B) Stability scatter ---
    ax = axes[1]
    sample_ids = np.arange(1, N + 1)
    ax.scatter(sample_ids, cos_sims, s=80, zorder=5,
               color="#59a14f", edgecolors="black", linewidth=0.7)
    mean_cos = results.get("stability_mean_cos", float("nan"))
    ax.axhline(mean_cos, ls="--", color="#e15759",
               lw=1.2, label=f"Mean = {mean_cos:.3f}")
    ax.axhline(0.8, ls=":", color="gray", lw=0.8, alpha=0.6, label="Threshold 0.8")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(sample_ids if N <= 20 else np.linspace(1, N, 5, dtype=int))
    ax.set_xlabel("Sequence #")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("(B) Stability\n(Gaussian noise perturbation)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5)

    # --- (C) Sparsity bars ---
    ax = axes[2]
    ax.bar([f"#{i+1}" for i in range(N)], gini_vals,
           color="#9c755f", edgecolor="black", linewidth=0.7)
    mean_gini = results.get("sparsity_gini", float("nan"))
    ax.axhline(mean_gini, ls="--", color="#e15759",
               lw=1.2, label=f"Mean = {mean_gini:.3f}")
    ax.axhline(0.5, ls=":", color="gray", lw=0.8, alpha=0.6,
               label="Sparse threshold 0.5")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Sequence")
    if N > 15:
        ax.set_xticks(np.linspace(0, N - 1, 5, dtype=int))
        ax.set_xticklabels([f"#{int(i)+1}" for i in np.linspace(0, N - 1, 5, dtype=int)])
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("(C) Sparsity\n(Gini of Grad-CAM)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5)

    # --- (D) Band localization horizontal bar ---
    ax = axes[3]
    bands = list(results["band_localization"].keys())
    fracs = [results["band_localization"][b] * 100 for b in bands]
    colors = [BAND_COLORS.get(b, "#4e79a7") for b in bands]
    hbars = ax.barh(bands, fracs, color=colors, edgecolor="black", linewidth=0.7)
    ax.set_xlabel("Attribution Mass (%)")
    ax.set_title("(D) Band Localization\n(mean % of CAM per band)",
                 fontsize=10, fontweight="bold")
    ax.bar_label(hbars, fmt="%.1f%%", fontsize=8, padding=3)
    ax.set_xlim(0, max(fracs) * 1.25 if fracs else 100)
    ax.invert_yaxis()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cnn_lstm_gradcam_band_attribution(all_heatmaps, sequences=None, true_labels=None,
                                           target_threshold=1, n_display=4,
                                           save_path="cnn_lstm_gradcam_band_attribution.png"):
    """Decompose CNN-LSTM Grad-CAM heatmaps into Delta, Theta, Alpha, and Beta bands across sequences.

    For each sequence, the temporal mean of the Grad-CAM heatmaps across frames is visualized
    and broken down into the 4 frequency bands with per-band energy bar charts.
    """
    all_heatmaps = np.asarray(all_heatmaps)
    N, T, H, W = all_heatmaps.shape
    N_plot = min(n_display, N) if n_display is not None else min(4, N)
    bands = list(get_band_rows(height=H).items())
    n_bands = len(bands)
    target_name = "class >= Drowsy" if target_threshold == 1 else "class >= Transition"

    freq_ticks = [_freq_to_row(f, height=H) for f in [0.5, 4, 8, 13, 30]]
    freq_labels = ["0.5", "4", "8", "13", "30"]

    all_band_energies = []
    for i in range(N):
        seq_cam = all_heatmaps[i].mean(axis=0)
        band_energies = {}
        for bname, (r0, r1) in bands:
            band_energies[bname] = float(seq_cam[r0:r1, :].mean())
        all_band_energies.append(band_energies)

    fig = plt.figure(figsize=(3.5 * (n_bands + 1), 3.0 * N_plot))
    outer = gridspec.GridSpec(N_plot, n_bands + 1, figure=fig, wspace=0.45, hspace=0.60)

    for i in range(N_plot):
        seq_cam = all_heatmaps[i].mean(axis=0)
        band_energies = all_band_energies[i]
        vmax_global = max(float(np.percentile(seq_cam, 99)), 1e-4)

        for b_idx, (bname, (r0, r1)) in enumerate(bands):
            masked = np.zeros_like(seq_cam)
            masked[r0:r1, :] = seq_cam[r0:r1, :]

            ax = fig.add_subplot(outer[i, b_idx])
            ax.imshow(masked, cmap="jet", aspect="auto", origin="lower", vmin=0, vmax=vmax_global)
            ax.set_title(bname, fontsize=8, fontweight="bold", color=BAND_COLORS.get(bname, "black"))
            ax.set_yticks(freq_ticks)
            ax.set_yticklabels(freq_labels, fontsize=6)
            ax.set_xticks([0, W // 2, W - 1])
            ax.set_xticklabels(["0s", "2.5s", "5s"], fontsize=6)
            ax.axhline(r0, color="white", lw=0.8, ls="--", alpha=0.7)
            ax.axhline(r1 - 1, color="white", lw=0.8, ls="--", alpha=0.7)

        # Bar chart column
        ax_bar = fig.add_subplot(outer[i, n_bands])
        names = list(band_energies.keys())
        vals = [band_energies[b] for b in names]
        colors = [BAND_COLORS.get(b, "#4e79a7") for b in names]
        bars = ax_bar.barh(names, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax_bar.set_xlabel("Mean CAM", fontsize=8)
        sample_title = f"Seq #{i+1} Band Energy"
        if true_labels is not None and i < len(true_labels):
            sample_title += f"\n(True: {true_labels[i]})"
        ax_bar.set_title(sample_title, fontsize=7.5, fontweight="bold")
        ax_bar.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=2)
        ax_bar.invert_yaxis()
        ax_bar.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        f"Per-Band CNN-LSTM Grad-CAM Attribution - {target_name} (Showing {N_plot}/{N} sequences)",
        fontsize=12, fontweight="bold", y=0.995
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-band attribution figure saved -> {save_path}")


def evaluate_cnn_lstm_gradcam_quantitative(model, data_source, true_labels=None,
                                           device="cuda", target_threshold=1,
                                           top_k_frac=0.10, noise_std=0.05,
                                           n_stability_reps=5, max_samples=30,
                                           save_path="cnn_lstm_gradcam_quantitative_evaluation.png",
                                           save_band_plot=True,
                                           band_save_path="cnn_lstm_gradcam_band_attribution.png"):
    """Compute Fidelity, Stability, Sparsity, and Band Localization for CNN-LSTM Grad-CAM.

    Parameters
    ----------
    model : torch.nn.Module
        Trained CNN-LSTM ordinal classifier.
    data_source : DataLoader, tuple, or torch.Tensor
        Validation or test data. Can be a PyTorch DataLoader yielding (sequences, labels),
        a tuple of (test_batch, labels), or a Tensor of shape (N, T, C, H, W).
    true_labels : array-like, optional
        Ground truth labels (if not provided inside data_source).
    device : str or torch.device
        Compute device.
    target_threshold : int
        Ordinal threshold index to explain: 0 means class >= Transition, 1 means class >= Drowsy.
    top_k_frac : float
        Fraction of spatio-temporal pixels to occlude for fidelity (default: 0.10).
    noise_std : float
        Standard deviation of Gaussian noise for stability (default: 0.05).
    n_stability_reps : int
        Number of noisy perturbation repetitions per sequence (default: 5).
    max_samples : int
        Maximum number of sequences to evaluate (default: 30).
    save_path : str
        Path where the 4-panel summary figure will be saved.

    Returns
    -------
    dict
        Dictionary containing fidelity_auc_drop, stability_mean_cos, sparsity_gini,
        and band_localization.
    """
    from sklearn.metrics import roc_auc_score

    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    model = model.to(device)
    model.eval()

    # Extract sequences and true_labels
    if isinstance(data_source, (tuple, list)) and len(data_source) == 2 and isinstance(data_source[0], torch.Tensor):
        sequences, lbls = data_source
        true_labels = lbls.cpu().numpy() if hasattr(lbls, "cpu") else np.array(lbls)
    elif isinstance(data_source, torch.utils.data.DataLoader):
        all_seqs, all_lbls = [], []
        for seqs, lbls in data_source:
            all_seqs.append(seqs)
            all_lbls.append(lbls)
            if sum(len(x) for x in all_seqs) >= max_samples:
                break
        sequences = torch.cat(all_seqs, dim=0)[:max_samples]
        true_labels = torch.cat(all_lbls, dim=0)[:max_samples].cpu().numpy()
    elif isinstance(data_source, torch.Tensor):
        sequences = data_source[:max_samples]
        if true_labels is not None:
            true_labels = true_labels.cpu().numpy() if hasattr(true_labels, "cpu") else np.array(true_labels)
            true_labels = true_labels[:max_samples]
    else:
        raise ValueError("data_source must be a DataLoader, (sequences, labels) tuple, or Tensor")

    N, T, C, H, W = sequences.shape
    if true_labels is None:
        true_labels = np.zeros(N, dtype=int)
    else:
        true_labels = np.asarray(true_labels)[:N]

    target_name = "class >= Drowsy" if target_threshold == 1 else "class >= Transition"
    print("\n" + "=" * 60)
    print("  QUANTITATIVE GRAD-CAM EVALUATION (CNN-LSTM)")
    print("=" * 60)
    print(f"  Sequences: {N} | Target threshold: {target_threshold} ({target_name})")

    # 1. Base Predictions & Grad-CAM generation
    all_heatmaps = []
    orig_scores = np.zeros(N)
    for i in range(N):
        seq_single = sequences[i].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(seq_single)
            cumulative = torch.sigmoid(logits)
            orig_scores[i] = cumulative[0, target_threshold].item()

        heatmaps, _, _ = generate_cnn_lstm_gradcam(
            model, seq_single, target_threshold=target_threshold, device=device
        )
        all_heatmaps.append(heatmaps)
    all_heatmaps = np.array(all_heatmaps)  # (N, T, H, W)

    # ------------------------------------------------------------------
    # A) FIDELITY (spatio-temporal pixel occlusion)
    # ------------------------------------------------------------------
    print("\n  [A] Fidelity (top-K spatio-temporal pixel occlusion)")
    occluded_scores = np.zeros(N)
    for i in range(N):
        attr_flat = all_heatmaps[i].flatten()
        k = max(1, int(top_k_frac * len(attr_flat)))
        top_idx = np.argsort(attr_flat)[::-1][:k]
        t_idx, r_idx, c_idx = np.unravel_index(top_idx, (T, H, W))

        seq_occ = sequences[i].clone()  # (T, C, H, W)
        ch_mean = seq_occ.mean(dim=[2, 3], keepdim=True)  # (T, C, 1, 1)
        for t, r, c in zip(t_idx, r_idx, c_idx):
            seq_occ[t, :, r, c] = ch_mean[t, :, 0, 0]

        with torch.no_grad():
            occ_logits = model(seq_occ.unsqueeze(0).to(device))
            occluded_scores[i] = torch.sigmoid(occ_logits)[0, target_threshold].item()

    binary_target = (true_labels >= target_threshold + 1).astype(int) if true_labels.max() > 1 else (true_labels == target_threshold).astype(int)
    try:
        if len(np.unique(binary_target)) > 1:
            auc_orig = float(roc_auc_score(binary_target, orig_scores))
            auc_occ = float(roc_auc_score(binary_target, occluded_scores))
            auc_drop = auc_orig - auc_occ
        else:
            auc_orig = float(np.mean(orig_scores))
            auc_occ = float(np.mean(occluded_scores))
            auc_drop = auc_orig - auc_occ
    except Exception:
        auc_orig = auc_occ = auc_drop = float("nan")

    print(f"    Original AUC/Score = {auc_orig:.4f}")
    print(f"    Occluded AUC/Score = {auc_occ:.4f}")
    fidelity_note = "good fidelity" if auc_drop > 0 else "low fidelity"
    print(f"    AUC drop           = {auc_drop:+.4f}  ({fidelity_note})")

    # ------------------------------------------------------------------
    # B) STABILITY (Gaussian noise perturbation)
    # ------------------------------------------------------------------
    print(f"\n  [B] Stability ({n_stability_reps} noisy perturbations, sigma={noise_std})")
    cos_sims = []
    for i in range(N):
        seq_single = sequences[i].unsqueeze(0)
        orig_cam = all_heatmaps[i]
        sims_i = []
        for _ in range(n_stability_reps):
            noisy_seq = seq_single + noise_std * torch.randn_like(seq_single)
            noisy_cam, _, _ = generate_cnn_lstm_gradcam(
                model, noisy_seq, target_threshold=target_threshold, device=device
            )
            sims_i.append(_cosine_sim(orig_cam, noisy_cam))
        mean_sim = float(np.mean(sims_i))
        cos_sims.append(mean_sim)
        if i < 5 or i == N - 1:
            print(f"    Sequence #{i+1}: mean cosine similarity = {mean_sim:.4f}")

    mean_stability = float(np.mean(cos_sims))
    stability_note = "stable" if mean_stability > 0.8 else "sensitive to noise"
    print(f"    Overall mean cos similarity = {mean_stability:.4f}  ({stability_note})")

    # ------------------------------------------------------------------
    # C) SPARSITY (Gini coefficient)
    # ------------------------------------------------------------------
    print("\n  [C] Sparsity (Gini coefficient)")
    gini_vals = [_gini(all_heatmaps[i]) for i in range(N)]
    mean_gini = float(np.mean(gini_vals))
    for i in range(min(5, N)):
        print(f"    Sequence #{i+1}: Gini = {gini_vals[i]:.4f}")
    sparsity_note = "sparse (focused)" if mean_gini > 0.5 else "diffuse attribution"
    print(f"    Mean Gini = {mean_gini:.4f}  ({sparsity_note})")

    # ------------------------------------------------------------------
    # D) FREQUENCY-BAND LOCALIZATION
    # ------------------------------------------------------------------
    print("\n  [D] Frequency-Band Localization")
    band_rows = get_band_rows(height=H)
    band_loc_all = {b: [] for b in band_rows}
    for i in range(N):
        sample_loc = {}
        for band_name, (r0, r1) in band_rows.items():
            sample_loc[band_name] = float(all_heatmaps[i, :, r0:r1, :].mean())
        total = sum(sample_loc.values()) + 1e-12
        for band_name in sample_loc:
            frac = sample_loc[band_name] / total
            band_loc_all[band_name].append(frac)

    band_localization = {k: float(np.mean(v)) for k, v in band_loc_all.items()}
    print("    Mean across sequences:")
    for band, frac in band_localization.items():
        bar = "|" * int(frac * 40)
        print(f"      {band:22s}: {frac * 100:5.1f}%  {bar}")

    results = {
        "fidelity_auc_drop": auc_drop,
        "fidelity_auc_orig": auc_orig,
        "fidelity_auc_occ": auc_occ,
        "stability_mean_cos": mean_stability,
        "sparsity_gini": mean_gini,
        "band_localization": band_localization,
    }

    # ------------------------------------------------------------------
    # 4-Panel Summary Figure & Per-Band Breakdown Figure
    # ------------------------------------------------------------------
    _plot_cnn_lstm_gradcam_quantitative_summary(
        results, gini_vals, cos_sims, auc_orig, auc_occ, N,
        title=f"Quantitative Grad-CAM Evaluation (CNN-LSTM: {target_name})",
        save_path=save_path
    )
    print(f"\n  Summary figure saved -> {save_path}")

    if save_band_plot:
        plot_cnn_lstm_gradcam_band_attribution(
            all_heatmaps, sequences=sequences, true_labels=true_labels,
            target_threshold=target_threshold, save_path=band_save_path
        )

    print("=" * 60)
    return results

