"""Grad-CAM explanation for the VGG16 EEG-spectrogram classifier.

Example usage (after ``trained_model`` and ``val_loader`` are created)::

    from Grad_CAM_Explainability import show_gradcam_example
    show_gradcam_example(trained_model, val_loader, device=device)

The function explains the class selected by the model unless ``target_class``
is supplied explicitly.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score


def generate_gradcam(model, image, target_layer=None, target_class=None,
                     device="cuda"):
    """Generate a Grad-CAM heatmap for one input spectrogram.

    Parameters
    ----------
    model : torch.nn.Module
        Trained image classifier.
    image : torch.Tensor
        One preprocessed image with shape ``(C, H, W)`` or ``(1, C, H, W)``.
    target_layer : torch.nn.Module, optional
        Convolutional layer to explain. For this VGG16 model, ``model.features[-1]``
        is a suitable default.
    target_class : int, optional
        Class to explain. If omitted, the model's predicted class is used.
    device : str or torch.device
        Device on which the model runs.

    Returns
    -------
    heatmap : np.ndarray
        Normalized Grad-CAM heatmap with values in ``[0, 1]``.
    predicted_class : int
        Model prediction.
    probabilities : np.ndarray
        Softmax probabilities for all classes.
    """
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    model = model.to(device)
    was_training = model.training
    model.eval()

    # VGG16 uses in-place ReLU operations. They can conflict with PyTorch's
    # backward hooks because the hook output is treated as a view. Disable
    # in-place behaviour only while computing Grad-CAM, then restore it.
    relu_layers = [layer for layer in model.modules() if isinstance(layer, nn.ReLU)]
    original_inplace = [layer.inplace for layer in relu_layers]
    for layer in relu_layers:
        layer.inplace = False

    if target_layer is None:
        # Use the final convolution, before the final ReLU, so the activation
        # map still has useful spatial information for the VGG16 classifier.
        target_layer = next(
            layer for layer in reversed(model.features)
            if isinstance(layer, nn.Conv2d)
        )

    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image.to(device).float()
    image.requires_grad_(True)

    activations = []
    gradients = []

    def save_activations(_module, _inputs, output):
        activations.append(output)

    def save_gradients(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(save_activations)
    backward_handle = target_layer.register_full_backward_hook(save_gradients)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(image)
        probabilities_tensor = torch.softmax(logits, dim=1)
        predicted_class = int(torch.argmax(probabilities_tensor, dim=1).item())
        class_to_explain = predicted_class if target_class is None else int(target_class)

        logits[0, class_to_explain].backward()

        # Average gradients over spatial locations to obtain channel weights.
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations[0]).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        probabilities = probabilities_tensor[0].detach().cpu().numpy()
        return cam, predicted_class, probabilities
    finally:
        forward_handle.remove()
        backward_handle.remove()
        image.requires_grad_(False)
        for layer, inplace in zip(relu_layers, original_inplace):
            layer.inplace = inplace
        if was_training:
            model.train()


def show_gradcam_example(model, data_loader, device="cuda", class_names=None,
                         target_class=None, sample_index=0,
                         save_path="gradcam_example.png"):
    """Display and save one validation-set Grad-CAM explanation.

    ``data_loader`` should yield ``(image, label)`` pairs from the existing
    ``DROZYSpectrogramDataset``. The displayed input is the normalized tensor;
    the heatmap remains valid, but the colors are intended for localization,
    not for recovering the original spectrogram color scale.
    """
    class_names = list(class_names or ["Alert", "Transition", "Drowsy"])
    if len(class_names) != int(model.classifier[-1].out_features):
        raise ValueError("class_names must have one name per model output class")
    images, labels = next(iter(data_loader))
    if sample_index >= len(images):
        raise IndexError(f"sample_index must be between 0 and {len(images) - 1}")

    image = images[sample_index]
    true_class = int(labels[sample_index].item())
    heatmap, predicted_class, probabilities = generate_gradcam(
        model, image, target_class=target_class, device=device
    )
    explained_class = predicted_class if target_class is None else int(target_class)

    # Convert the 3-channel normalized tensor to a displayable grayscale image.
    # Averaging channels is appropriate here because the EEG channels are
    # encoded as image channels (for example Fz, Cz, and C3).
    display_image = image.detach().cpu().numpy().transpose(1, 2, 0).mean(axis=2)
    display_image -= display_image.min()
    display_image /= display_image.max() + 1e-8

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(display_image, cmap="viridis", origin="lower", aspect="auto")
    axes[0].set_title(f"Input spectrogram\nTrue: {class_names[true_class]}")

    axes[1].imshow(heatmap, cmap="jet", origin="lower", aspect="auto")
    axes[1].set_title("Grad-CAM importance")

    axes[2].imshow(display_image, cmap="gray", origin="lower", aspect="auto")
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45, origin="lower", aspect="auto")
    axes[2].set_title(
        f"Explained: {class_names[explained_class]}\n"
        f"Prediction: {class_names[predicted_class]} "
        f"({probabilities[predicted_class]:.2%})\n"
        f"Explained score: {probabilities[explained_class]:.2%}"
    )

    for axis in axes:
        axis.set_xlabel("Time")
        axis.set_ylabel("Frequency")
    fig.suptitle("Grad-CAM Explanation for EEG Drowsiness Detection")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return heatmap, predicted_class, probabilities


# ---------------------------------------------------------------------------
# Quantitative Grad-CAM Quality Metrics (VGG16)
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
    """Cosine similarity between two attribution maps."""
    a, b = a.flatten(), b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def _plot_gradcam_quantitative_summary(results, gini_vals, cos_sims,
                                       auc_orig, auc_occ, N,
                                       title="Quantitative Grad-CAM Evaluation (VGG16)",
                                       save_path="gradcam_quantitative_evaluation.png"):
    """Produces a 4-panel standalone summary figure for quantitative Grad-CAM metrics."""
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
    ax.set_ylabel("AUC / Prediction Score")
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
    ax.set_xlabel("Sample #")
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
    ax.set_xlabel("Sample")
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


def plot_gradcam_band_attribution(cams, test_images=None, true_labels=None,
                                  class_idx=2, class_names=None, n_display=4,
                                  save_path="gradcam_band_attribution.png"):
    """Decompose Grad-CAM heatmaps into Delta, Theta, Alpha, and Beta frequency bands.

    Parameters
    ----------
    cams : np.ndarray or list
        Grad-CAM heatmaps with shape (N, H, W).
    test_images : torch.Tensor or np.ndarray, optional
        Test spectrograms.
    true_labels : array-like, optional
        Ground truth labels for annotation.
    class_idx : int
        Class index being explained.
    class_names : list of str, optional
        Class names list.
    n_display : int
        Number of samples to visualize.
    save_path : str
        Output file path.
    """
    cams = np.asarray(cams)
    N = len(cams)
    H, W = cams.shape[-2], cams.shape[-1]
    N_plot = min(n_display, N) if n_display is not None else min(4, N)
    bands = list(get_band_rows(height=H).items())
    n_bands = len(bands)

    if class_names is None:
        class_names = ["Alert", "Transition", "Drowsy"]
    class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"

    freq_ticks = [_freq_to_row(f, height=H) for f in [0.5, 4, 8, 13, 30]]
    freq_labels = ["0.5", "4", "8", "13", "30"]

    all_band_energies = []
    for i in range(N):
        cam = cams[i]
        band_energies = {}
        for bname, (r0, r1) in bands:
            band_energies[bname] = float(cam[r0:r1, :].mean())
        all_band_energies.append(band_energies)

    fig = plt.figure(figsize=(3.5 * (n_bands + 1), 3.0 * N_plot))
    outer = gridspec.GridSpec(N_plot, n_bands + 1, figure=fig, wspace=0.45, hspace=0.60)

    for i in range(N_plot):
        cam = cams[i]
        band_energies = all_band_energies[i]
        vmax_global = max(float(np.percentile(cam, 99)), 1e-4)

        for b_idx, (bname, (r0, r1)) in enumerate(bands):
            masked = np.zeros_like(cam)
            masked[r0:r1, :] = cam[r0:r1, :]

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
        sample_title = f"#{i+1} Band Energy"
        if true_labels is not None and i < len(true_labels):
            t_lbl = class_names[int(true_labels[i])] if int(true_labels[i]) < len(class_names) else str(true_labels[i])
            sample_title += f"\n(True: {t_lbl})"
        ax_bar.set_title(sample_title, fontsize=7.5, fontweight="bold")
        ax_bar.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=2)
        ax_bar.invert_yaxis()
        ax_bar.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        f"Per-Band Grad-CAM Attribution - {class_name} (Showing {N_plot}/{N} samples)",
        fontsize=12, fontweight="bold", y=0.995
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-band attribution figure saved -> {save_path}")


def evaluate_gradcam_quantitative(model, data_source, true_labels=None,
                                  device="cuda", target_class=None,
                                  top_k_frac=0.10, noise_std=0.05,
                                  n_stability_reps=5, max_samples=40,
                                  save_path="gradcam_quantitative_evaluation.png",
                                  save_band_plot=True,
                                  band_save_path="gradcam_band_attribution.png"):
    """Compute Fidelity, Stability, Sparsity, and Band Localization for VGG16 Grad-CAM.

    Parameters
    ----------
    model : torch.nn.Module
        Trained VGG16 classifier.
    data_source : DataLoader, tuple, or torch.Tensor
        Validation or test data. Can be a PyTorch DataLoader yielding (images, labels),
        a tuple of (test_batch, labels), or a Tensor of shape (N, C, H, W).
    true_labels : array-like, optional
        Ground truth labels (if not provided inside data_source).
    device : str or torch.device
        Compute device.
    target_class : int, optional
        Target class index to explain (default is class-2 / Drowsy, or highest available).
    top_k_frac : float
        Fraction of pixels to occlude for fidelity evaluation (default: 0.10).
    noise_std : float
        Standard deviation of Gaussian noise for stability evaluation (default: 0.05).
    n_stability_reps : int
        Number of noisy perturbation repetitions per sample (default: 5).
    max_samples : int
        Maximum number of samples to evaluate (default: 40).
    save_path : str
        Path where the 4-panel summary figure will be saved.

    Returns
    -------
    dict
        Dictionary containing fidelity_auc_drop, stability_mean_cos, sparsity_gini,
        and band_localization.
    """
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    model = model.to(device)
    model.eval()

    # Extract test_batch and true_labels from data_source
    if isinstance(data_source, (tuple, list)) and len(data_source) == 2 and isinstance(data_source[0], torch.Tensor):
        test_batch, lbls = data_source
        true_labels = lbls.cpu().numpy() if hasattr(lbls, "cpu") else np.array(lbls)
    elif isinstance(data_source, torch.utils.data.DataLoader):
        all_imgs, all_lbls = [], []
        for imgs, lbls in data_source:
            all_imgs.append(imgs)
            all_lbls.append(lbls)
            if sum(len(x) for x in all_imgs) >= max_samples:
                break
        test_batch = torch.cat(all_imgs, dim=0)[:max_samples]
        true_labels = torch.cat(all_lbls, dim=0)[:max_samples].cpu().numpy()
    elif isinstance(data_source, torch.Tensor):
        test_batch = data_source[:max_samples]
        if true_labels is not None:
            true_labels = true_labels.cpu().numpy() if hasattr(true_labels, "cpu") else np.array(true_labels)
            true_labels = true_labels[:max_samples]
    else:
        raise ValueError("data_source must be a DataLoader, (batch, labels) tuple, or Tensor")

    N, C, H, W = test_batch.shape
    if true_labels is None:
        true_labels = np.zeros(N, dtype=int)
    else:
        true_labels = np.asarray(true_labels)[:N]

    out_features = int(model.classifier[-1].out_features)
    is_ordinal = (out_features == 2 and true_labels.max() > 1)
    if target_class is None:
        target_class = out_features - 1
    else:
        target_class = min(int(target_class), out_features - 1)

    print("\n" + "=" * 60)
    print("  QUANTITATIVE GRAD-CAM EVALUATION (VGG16)")
    print("=" * 60)
    print(f"  Samples: {N} | Target class/threshold: {target_class}")

    # 1. Base Predictions & Grad-CAM generation
    cams = []
    orig_scores = np.zeros(N)
    with torch.no_grad():
        out = model(test_batch.to(device))
        if out_features > 1 and not is_ordinal:
            probs = torch.softmax(out, dim=1).cpu().numpy()
            orig_scores = probs[:, target_class]
        else:
            probs = torch.sigmoid(out).cpu().numpy()
            orig_scores = probs[:, target_class]

    for i in range(N):
        cam, _, _ = generate_gradcam(
            model, test_batch[i].unsqueeze(0), target_class=target_class, device=device
        )
        cams.append(cam)
    cams = np.array(cams)

    # ------------------------------------------------------------------
    # A) FIDELITY (top-K pixel occlusion)
    # ------------------------------------------------------------------
    print("\n  [A] Fidelity (top-K pixel occlusion)")
    occluded_scores = np.zeros(N)
    for i in range(N):
        attr_flat = cams[i].flatten()
        k = max(1, int(top_k_frac * len(attr_flat)))
        top_idx = np.argsort(attr_flat)[::-1][:k]
        top_2d = np.unravel_index(top_idx, (H, W))

        inp_occ = test_batch[i].clone()
        ch_mean = inp_occ.mean(dim=[1, 2], keepdim=True)
        for r, c in zip(top_2d[0], top_2d[1]):
            inp_occ[:, r, c] = ch_mean[:, 0, 0]

        with torch.no_grad():
            out_occ = model(inp_occ.unsqueeze(0).to(device))
            if out_features > 1 and not is_ordinal:
                occ_p = torch.softmax(out_occ, dim=1).cpu().numpy()[0, target_class]
            else:
                occ_p = torch.sigmoid(out_occ).cpu().numpy()[0, target_class]
            occluded_scores[i] = occ_p

    binary_target = (true_labels >= target_class + 1).astype(int) if is_ordinal else (true_labels == target_class).astype(int)
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
        inp_single = test_batch[i].unsqueeze(0)
        orig_cam = cams[i]
        sims_i = []
        for _ in range(n_stability_reps):
            noisy_inp = inp_single + noise_std * torch.randn_like(inp_single)
            noisy_cam, _, _ = generate_gradcam(
                model, noisy_inp, target_class=target_class, device=device
            )
            sims_i.append(_cosine_sim(orig_cam, noisy_cam))
        mean_sim = float(np.mean(sims_i))
        cos_sims.append(mean_sim)
        if i < 5 or i == N - 1:
            print(f"    Sample #{i+1}: mean cosine similarity = {mean_sim:.4f}")

    mean_stability = float(np.mean(cos_sims))
    stability_note = "stable" if mean_stability > 0.8 else "sensitive to noise"
    print(f"    Overall mean cos similarity = {mean_stability:.4f}  ({stability_note})")

    # ------------------------------------------------------------------
    # C) SPARSITY (Gini coefficient)
    # ------------------------------------------------------------------
    print("\n  [C] Sparsity (Gini coefficient)")
    gini_vals = [_gini(cams[i]) for i in range(N)]
    mean_gini = float(np.mean(gini_vals))
    for i in range(min(5, N)):
        print(f"    Sample #{i+1}: Gini = {gini_vals[i]:.4f}")
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
            sample_loc[band_name] = float(cams[i, r0:r1, :].mean())
        total = sum(sample_loc.values()) + 1e-12
        for band_name in sample_loc:
            frac = sample_loc[band_name] / total
            band_loc_all[band_name].append(frac)

    band_localization = {k: float(np.mean(v)) for k, v in band_loc_all.items()}
    print("    Mean across samples:")
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
    _plot_gradcam_quantitative_summary(
        results, gini_vals, cos_sims, auc_orig, auc_occ, N,
        title="Quantitative Grad-CAM Evaluation (VGG16)",
        save_path=save_path
    )
    print(f"\n  Summary figure saved -> {save_path}")

    if save_band_plot:
        plot_gradcam_band_attribution(
            cams, test_images=test_batch, true_labels=true_labels,
            class_idx=target_class, save_path=band_save_path
        )

    print("=" * 60)
    return results


