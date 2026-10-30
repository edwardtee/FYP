"""
SHAP_Analysis_And_Evaluation.py
================================
Addresses three critical SHAP review findings:

  1. SHAP Output Structure Verification
     - Side-by-side Alert vs Drowsy attributions with sign-flip correlation check.

  2. Per-Band Attribution Breakdown
     - Isolates SHAP energy in Delta (0.5-4 Hz), Theta (4-8 Hz),
       Alpha (8-13 Hz), and Beta (13-30 Hz) sub-bands.

  3. Quantitative Evaluation
     - Fidelity    : AUC-drop when high-attribution pixels are occluded.
     - Stability   : Mean cosine similarity between SHAP maps of near-identical
                     inputs (Gaussian noise perturbation).
     - Sparsity    : Gini coefficient of the absolute attribution distribution.
     - Band Localization : Fraction of total attribution mass in each EEG band.

Usage (from Execution_Pipeline.py):
    from SHAP_Analysis_And_Evaluation import run_shap_analysis
    run_shap_analysis(trained_model, train_loader, val_loader, device=device)
"""

import shap
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# np.flipud is NO LONGER applied in STFT_Spectrogram_Generation.
# scipy.signal.spectrogram returns frequencies in ascending order, so:
#   row 0 = lowest frequency (0.5 Hz), row 223 = highest frequency (30 Hz).
# Displayed with origin='lower', this gives the standard convention:
# low freq at the bottom and high freq at the top.
FREQ_MIN_HZ, FREQ_MAX_HZ, IMAGE_HEIGHT = 0.5, 30.0, 224

def _freq_to_row(freq_hz):
    """Convert Hz to row index in the natural (unflipped) image (row 0 = 0.5 Hz bottom)."""
    frac = (freq_hz - FREQ_MIN_HZ) / (FREQ_MAX_HZ - FREQ_MIN_HZ)
    return int(round(frac * IMAGE_HEIGHT))

# Band row ranges in the natural (unflipped) image coordinate system.
# Row 0 = 0.5 Hz (bottom), row 224 = 30 Hz (top) when displayed with origin='lower'.
BAND_ROWS = {
    "Delta (0.5-4 Hz)":  (0,                 _freq_to_row(4)),
    "Theta (4-8 Hz)":    (_freq_to_row(4),   _freq_to_row(8)),
    "Alpha (8-13 Hz)":   (_freq_to_row(8),   _freq_to_row(13)),
    "Beta  (13-30 Hz)":  (_freq_to_row(13),  IMAGE_HEIGHT),
}
BAND_COLORS = {
    "Delta (0.5-4 Hz)":  "#4e79a7",
    "Theta (4-8 Hz)":    "#f28e2b",
    "Alpha (8-13 Hz)":   "#59a14f",
    "Beta  (13-30 Hz)":  "#e15759",
}


def _extract_shap_for_class(shap_values, class_idx, num_classes):
    """
    Robustly extract a (N, H, W) float32 attribution map for *class_idx*
    from whatever structure SHAP GradientExplainer returns.
    Returns a numpy array of shape (N, H, W).
    """
    def _to_np(x):
        return x.cpu().detach().numpy() if hasattr(x, "cpu") else np.array(x)

    sv = _to_np(shap_values) if not isinstance(shap_values, list) else None

    # List of per-class arrays
    if isinstance(shap_values, list):
        arr = _to_np(shap_values[class_idx])   # (N, C, H, W) or (N, H, W, C)
    elif sv.ndim == 5:
        if sv.shape[-1] == num_classes:
            arr = sv[..., class_idx]            # (N, C, H, W)
        else:
            arr = sv[class_idx]                 # (N, C, H, W)
    else:
        arr = sv                                # single class

    # Channels-first -> channels-last
    if arr.ndim == 4 and arr.shape[1] in (1, 3, 4):
        arr = np.transpose(arr, (0, 2, 3, 1))  # (N, H, W, C)

    # Collapse colour channels to single attribution map
    if arr.ndim == 4:
        arr = arr.mean(axis=-1)                 # (N, H, W)

    return arr.astype(np.float32)


def _gini(values):
    """Gini coefficient of a flat array (higher -> sparser)."""
    v = np.abs(values.flatten())
    v = np.sort(v)
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def _cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# 1.  SHAP Structure Verification
# ---------------------------------------------------------------------------

def verify_shap_structure(shap_values, num_classes, test_images_np,
                           true_labels, pred_labels, probs, n_display=None):
    """
    Plots Alert (class-0) and Drowsy (class-1) attribution maps side-by-side
    and reports the Pearson r between them.

    Parameters
    ----------
    n_display : int or None
        Number of samples to include in the *figure* (first n_display rows).
        Console stats are always printed for all N samples.
        Defaults to None (show all).

    The model returns logits, therefore correlation between class maps is a
    descriptive diagnostic only; it is not a validity criterion.
    """
    alert_maps  = _extract_shap_for_class(shap_values, 0, num_classes)
    drowsy_maps = _extract_shap_for_class(shap_values, min(2, num_classes - 1), num_classes)
    N      = len(test_images_np)
    N_plot = min(n_display, N) if n_display is not None else N

    print("\n" + "="*60)
    print("  [1] SHAP STRUCTURE VERIFICATION")
    print(f"  (stats: all {N} samples | figure: first {N_plot} samples)")
    print("="*60)

    # --- Console stats for ALL N samples ---
    class_names = ["Alert", "Transition", "Drowsy"][:num_classes]
    for i in range(N):
        r, pval = pearsonr(alert_maps[i].flatten(), drowsy_maps[i].flatten())
        sign_str = ("anti-correlated" if r < -0.7 else "not strongly anti-correlated")
        print(f"  Sample #{i+1}: Pearson r(alert, drowsy) = {r:+.4f}  p={pval:.3e}  -> {sign_str}")

    # --- Figure for first N_plot samples only ---
    freq_ticks  = [_freq_to_row(f) for f in [0.5, 4, 8, 13, 30]]
    freq_labels = ["0.5", "4", "8", "13", "30"]
    time_ticks  = [0, 56, 112, 168, 224]
    time_labels = ["0s", "1.25s", "2.5s", "3.75s", "5s"]

    fig, axes = plt.subplots(N_plot, 3, figsize=(15.5, 8.5),
                             gridspec_kw={"wspace": 0.45, "hspace": 0.80})
    if N_plot == 1:
        axes = axes[np.newaxis, :]

    for i in range(N_plot):
        # ---- Input spectrogram ----
        spec = test_images_np[i].mean(axis=-1) if test_images_np[i].ndim == 3 else test_images_np[i]
        im0 = axes[i, 0].imshow(spec, cmap="viridis", aspect="auto", origin="lower")
        t_str = class_names[int(true_labels[i])]
        p_str = class_names[int(pred_labels[i])]
        conf  = probs[i, int(pred_labels[i])] * 100
        col   = "darkgreen" if true_labels[i] == pred_labels[i] else "darkred"
        axes[i, 0].set_title(f"#{i+1}  True:{t_str} | Pred:{p_str} ({conf:.1f}%)",
                             fontsize=8, fontweight="bold", color=col)
        cbar0 = fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        cbar0.ax.tick_params(labelsize=6)

        vmax_a = np.percentile(np.abs(alert_maps[i]),  99) + 1e-8
        vmax_d = np.percentile(np.abs(drowsy_maps[i]), 99) + 1e-8

        # ---- Alert attribution ----
        im1 = axes[i, 1].imshow(alert_maps[i],  cmap="seismic", aspect="auto",
                                origin="lower", vmin=-vmax_a, vmax=vmax_a)
        axes[i, 1].set_title("Alert (class-0) Attribution", fontsize=8.5)
        cbar1 = fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=6)
        
        # ---- Drowsy attribution ----
        im2 = axes[i, 2].imshow(drowsy_maps[i], cmap="seismic", aspect="auto",
                                origin="lower", vmin=-vmax_d, vmax=vmax_d)
        axes[i, 2].set_title(
            f"{class_names[min(2, num_classes - 1)]} Attribution", fontsize=8.5
        )
        cbar2 = fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=6)

        # Pearson r (re-computed for label annotation)
        r, _ = pearsonr(alert_maps[i].flatten(), drowsy_maps[i].flatten())
        sign_str = ("anti-correlated" if r < -0.7 else "not strongly anti-correlated")
        axes[i, 2].set_xlabel(f"r = {r:.3f}  {sign_str}", fontsize=7.5, color="navy")

        # Tick formatting
        for col_ax in axes[i]:
            col_ax.set_yticks(freq_ticks)
            col_ax.set_yticklabels(freq_labels, fontsize=6)
            col_ax.set_xticks(time_ticks)
            col_ax.set_xticklabels(time_labels, fontsize=6)
            col_ax.set_ylabel("Freq (Hz)", fontsize=7)

    fig.suptitle(
        f"SHAP Structure Check - Alert vs Drowsy Attribution Maps"
        f"  (showing {N_plot}/{N} samples)",
        fontsize=11, fontweight="bold", y=0.995)
    plt.savefig("shap_structure_verification.png", dpi=150, bbox_inches="tight")
    #plt.show()
    print("  Figure saved -> shap_structure_verification.png\n")


# ---------------------------------------------------------------------------
# 2.  Per-Band Attribution
# ---------------------------------------------------------------------------

def plot_band_attribution(shap_values, num_classes, test_images_np,
                           true_labels, pred_labels, class_idx=1, n_display=None):
    """
    Decomposes the SHAP map into Delta / Theta / Alpha / Beta frequency bands
    and shows:
      - A column per band (masked attribution heat-map).
      - A bar chart of total attribution energy per band.

    Parameters
    ----------
    n_display : int or None
        Number of samples to include in the *figure* (first n_display rows).
        Console stats are always printed for all N samples.
        Defaults to None (show all).
    """
    maps       = _extract_shap_for_class(shap_values, class_idx, num_classes)
    N          = len(test_images_np)
    N_plot     = min(n_display, N) if n_display is not None else N
    bands      = list(BAND_ROWS.items())   # [(name, (r0, r1)), ...]
    n_bands    = len(bands)
    class_names = ["Alert", "Transition", "Drowsy"][:num_classes]
    class_name = class_names[class_idx]

    print("\n" + "="*60)
    print(f"  [2] PER-BAND ATTRIBUTION  (class: {class_name})")
    print(f"  (stats: all {N} samples | figure: first {N_plot} samples)")
    print("="*60)

    freq_ticks  = [_freq_to_row(f) for f in [0.5, 4, 8, 13, 30]]
    freq_labels = ["0.5", "4", "8", "13", "30"]
    # --- Console stats for ALL N samples ---
    all_band_energies = []
    for i in range(N):
        attr = maps[i]   # (H, W)
        band_energies = {}
        for band_name, (r0, r1) in bands:
            masked_slice = attr[r0:r1, :]
            band_energies[band_name] = float(np.abs(masked_slice).mean())
        all_band_energies.append(band_energies)

        t_str = class_names[int(true_labels[i])]
        p_str = class_names[int(pred_labels[i])]
        print(f"\n  Sample #{i+1}  (True:{t_str} | Pred:{p_str})")
        vals  = list(band_energies.values())
        names = list(band_energies.keys())
        total = sum(vals) + 1e-12
        for bname, bval in zip(names, vals):
            print(f"    {bname:22s}: {bval:.5f}  ({bval/total*100:.1f}%)")
        dominant = names[int(np.argmax(vals))]
        print(f"    -> Dominant band: {dominant}")

    # --- Figure for first N_plot samples only ---
    fig = plt.figure(figsize=(3.5 * (n_bands + 1), 3.2 * N_plot))
    outer = gridspec.GridSpec(N_plot, n_bands + 1, figure=fig,
                              wspace=0.45, hspace=0.85)

    for i in range(N_plot):
        attr          = maps[i]   # (H, W)
        band_energies = all_band_energies[i]
        vmax_global   = np.percentile(np.abs(attr), 99) + 1e-8

        for b_idx, (band_name, (r0, r1)) in enumerate(bands):
            masked = np.zeros_like(attr)
            masked[r0:r1, :] = attr[r0:r1, :]

            ax = fig.add_subplot(outer[i, b_idx])
            ax.imshow(masked, cmap="seismic", aspect="auto",
                      origin="lower", vmin=-vmax_global, vmax=vmax_global)
            ax.set_title(band_name, fontsize=7.5, fontweight="bold",
                         color=BAND_COLORS[band_name])
            ax.set_yticks(freq_ticks)
            ax.set_yticklabels(freq_labels, fontsize=5.5)
            ax.set_xticks([0, 112, 224])
            ax.set_xticklabels(["0s", "2.5s", "5s"], fontsize=5.5)
            ax.axhline(r0, color="white", lw=0.8, ls="--", alpha=0.7)
            ax.axhline(r1 - 1, color="white", lw=0.8, ls="--", alpha=0.7)

        # Bar chart column
        ax_bar = fig.add_subplot(outer[i, n_bands])
        names  = list(band_energies.keys())
        vals   = [band_energies[b] for b in names]
        colors = [BAND_COLORS[b] for b in names]
        bars   = ax_bar.barh(names, vals, color=colors,
                             edgecolor="black", linewidth=0.6)
        ax_bar.set_xlabel("Mean |SHAP|", fontsize=8)
        ax_bar.set_title(f"#{i+1} Band Energy", fontsize=8, fontweight="bold")
        ax_bar.bar_label(bars, fmt="%.4f", fontsize=6.5, padding=2)
        ax_bar.invert_yaxis()
        ax_bar.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        f"Per-Band SHAP Attribution - {class_name} Class"
        f"  (showing {N_plot}/{N} samples)",
        fontsize=12, fontweight="bold", y=0.995)
    plt.savefig("shap_band_attribution.png", dpi=150, bbox_inches="tight")
    #plt.show()
    print("\n  Figure saved -> shap_band_attribution.png\n")


# ---------------------------------------------------------------------------
# 3.  Quantitative Evaluation
# ---------------------------------------------------------------------------

def evaluate_shap_quantitative(model, test_batch, shap_values, num_classes,
                                true_labels, probs, background_batch, device="cuda",
                                target_class=None,
                                top_k_frac=0.10, noise_std=0.05,
                                n_stability_reps=5, shap_nsamples=50):
    """
    Computes four quantitative SHAP quality metrics.

    Fidelity
    --------
    Occludes the top-K% highest-attribution pixels with the channel mean,
    runs the model on the degraded input, and reports the AUC drop vs the
    original prediction.  A larger drop indicates higher fidelity.
    K is set by *top_k_frac* (default 0.10 = top 10% of pixels).

    Stability
    ---------
    Generates `n_stability_reps` slightly noisy copies of each test input,
    recomputes SHAP for each copy, and measures mean cosine similarity between
    the original and noisy attribution maps.  Values close to 1.0 are ideal or greater than 0.8.
    It measures whether the explanation remains similar when the input image is changed slightly.

    Sparsity
    --------
    Gini coefficient of the absolute SHAP values (0 = uniform, 1 = maximally
    sparse).  Higher values suggest a tighter, more interpretable explanation.
    It measures how concentrated the absolute SHAP attribution values are across the image.

    Frequency-Band Localization
    ---------------------------
    Fraction of total |SHAP| mass residing in each EEG frequency band.
    A well-formed EEG explainer should concentrate in biologically meaningful
    bands (Theta / Alpha are classically associated with drowsiness).

    Parameters
    ----------
    model             : trained PyTorch model (eval mode assumed)
    test_batch        : torch.Tensor  (N, C, H, W) on device
    shap_values       : raw output of GradientExplainer.shap_values()
    num_classes       : int
    true_labels       : (N,) numpy array of ground-truth class indices
    probs             : (N, num_classes) softmax probabilities
    top_k_frac        : fraction of pixels to occlude for fidelity test
    noise_std         : Gaussian noise std for stability test
    n_stability_reps  : number of noisy copies per sample (stability)

    Returns
    -------
    dict with keys: fidelity_auc_drop, stability_mean_cos,
                    sparsity_gini, band_localization
    """
    class_names = ["Alert", "Transition", "Drowsy"][:num_classes]
    results = {}

    # For VLA_VRW, use Drowsy (class-2) as the primary one-vs-rest target.
    # The explicit parameter also allows Transition or Alert to be studied.
    target_class = min(2, num_classes - 1) if target_class is None else int(target_class)
    maps = _extract_shap_for_class(shap_values, target_class, num_classes)

    print("\n" + "="*60)
    print("  [3] QUANTITATIVE SHAP EVALUATION")
    print("="*60)

    # ------------------------------------------------------------------
    # A) FIDELITY  (occlusion-based AUC drop)
    # ------------------------------------------------------------------
    print("\n  [A] Fidelity (top-K pixel occlusion)")
    model.eval()
    N, C, H, W = test_batch.shape

    orig_scores = probs[:, target_class]

    occluded_scores = np.zeros(N)
    for i in range(N):
        attr_flat = maps[i].flatten()   # (H*W,)
        k = max(1, int(top_k_frac * len(attr_flat)))
        top_idx  = np.argsort(np.abs(attr_flat))[::-1][:k]
        top_2d   = np.unravel_index(top_idx, (H, W))

        inp = test_batch[i].clone()   # (C, H, W)
        ch_mean = inp.mean(dim=[1, 2], keepdim=True)
        for r, c in zip(top_2d[0], top_2d[1]):
            inp[:, r, c] = ch_mean[:, 0, 0]

        with torch.no_grad():
            out = model(inp.unsqueeze(0).to(device))
            occ_prob = torch.softmax(out, dim=1).cpu().numpy()[0, target_class]
        occluded_scores[i] = occ_prob

    try:
        binary_target = (true_labels == target_class).astype(np.int8)
        auc_orig = roc_auc_score(binary_target, orig_scores)
        auc_occ  = roc_auc_score(binary_target, occluded_scores)
        auc_drop = auc_orig - auc_occ
    except ValueError:
        auc_orig = auc_occ = auc_drop = float("nan")

    results["fidelity_auc_drop"] = auc_drop
    print(f"    AUC (original)  = {auc_orig:.4f}")
    print(f"    AUC (occluded)  = {auc_occ:.4f}")
    fidelity_note = "good fidelity" if auc_drop > 0 else "low fidelity - check explainer"
    print(f"    AUC drop        = {auc_drop:+.4f}  ({fidelity_note})")

    # ------------------------------------------------------------------
    # B) STABILITY  (cosine similarity under Gaussian noise)
    # ------------------------------------------------------------------
    print(f"\n  [B] Stability ({n_stability_reps} noisy perturbations, sigma={noise_std})")
    # Use exactly the same training background as the original explanations.
    # Using the explained test samples as a new baseline makes this comparison
    # measure a different SHAP game and invalidates the stability estimate.
    explainer_bg = shap.GradientExplainer(model, background_batch)

    cos_sims = []
    for i in range(N):
        # Compute the reference map separately with the same seed as every
        # perturbation.  The previous comparison used the i-th map from a
        # batch run, whose Monte-Carlo sequence differs from a one-sample run.
        # That inflated apparent instability.
        original_sv = explainer_bg.shap_values(
            test_batch[i].unsqueeze(0).to(device),
            nsamples=shap_nsamples, rseed=42
        )
        orig_map = _extract_shap_for_class(original_sv, target_class, num_classes)[0]
        sims_i   = []
        for _ in range(n_stability_reps):
            noisy_inp = (test_batch[i] + noise_std * torch.randn_like(test_batch[i])).unsqueeze(0)
            # Same Monte-Carlo seed isolates sensitivity to the input change,
            # rather than measuring GradientExplainer sampling noise.
            noisy_sv = explainer_bg.shap_values(
                noisy_inp.to(device), nsamples=shap_nsamples, rseed=42
            )
            noisy_map = _extract_shap_for_class(noisy_sv, target_class, num_classes)[0]
            sims_i.append(_cosine_sim(orig_map, noisy_map))
        mean_sim = float(np.mean(sims_i))
        cos_sims.append(mean_sim)
        print(f"    Sample #{i+1}: mean cosine similarity = {mean_sim:.4f}")

    results["stability_mean_cos"] = float(np.mean(cos_sims))
    stability_note = "stable" if results["stability_mean_cos"] > 0.8 else "unstable - consider more background samples"
    print(f"    Overall mean cos similarity = {results['stability_mean_cos']:.4f}  ({stability_note})")

    # ------------------------------------------------------------------
    # C) SPARSITY  (Gini coefficient)
    # ------------------------------------------------------------------
    print("\n  [C] Sparsity (Gini coefficient)")
    gini_vals = [_gini(maps[i]) for i in range(N)]
    results["sparsity_gini"] = float(np.mean(gini_vals))
    for i, g in enumerate(gini_vals):
        print(f"    Sample #{i+1}: Gini = {g:.4f}")
    sparsity_note = "sparse (focused)" if results["sparsity_gini"] > 0.5 else "diffuse attribution"
    print(f"    Mean Gini = {results['sparsity_gini']:.4f}  ({sparsity_note})")

    # ------------------------------------------------------------------
    # D) FREQUENCY-BAND LOCALIZATION
    # ------------------------------------------------------------------
    print("\n  [D] Frequency-Band Localization")
    band_loc_all = {b: [] for b in BAND_ROWS}
    for i in range(N):
        # Compare mean absolute attribution per pixel, not raw mass.  Beta has
        # many more image rows than delta and would otherwise be favoured solely
        # by its larger area.
        sample_loc = {}
        for band_name, (r0, r1) in BAND_ROWS.items():
            sample_loc[band_name] = float(np.abs(maps[i][r0:r1, :]).mean())
        total = sum(sample_loc.values()) + 1e-12
        for band_name in sample_loc:
            frac = sample_loc[band_name] / total
            band_loc_all[band_name].append(frac)
            sample_loc[band_name] = frac
        print(f"    Sample #{i+1}: " +
              "  ".join(f"{k.split()[0]}={v*100:.1f}%" for k, v in sample_loc.items()))

    results["band_localization"] = {k: float(np.mean(v)) for k, v in band_loc_all.items()}
    print("    Mean across samples:")
    for band, frac in results["band_localization"].items():
        bar = "|" * int(frac * 40)
        print(f"      {band:22s}: {frac*100:5.1f}%  {bar}")

    # ------------------------------------------------------------------
    # Summary figure
    # ------------------------------------------------------------------
    _plot_quantitative_summary(results, gini_vals, cos_sims,
                               auc_orig, auc_occ, N)

    print("\n  Figure saved -> shap_quantitative_evaluation.png")
    return results


def _plot_quantitative_summary(results, gini_vals, cos_sims,
                               auc_orig, auc_occ, N):
    """Produces a 4-panel summary figure for the quantitative metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5),
                             gridspec_kw={"wspace": 0.45})
    fig.suptitle("Quantitative SHAP Evaluation", fontsize=13,
                 fontweight="bold", y=1.02)

    # --- (A) Fidelity bar ---
    ax = axes[0]
    vals_a   = [auc_orig, auc_occ]
    colors_a = ["#4e79a7", "#f28e2b"]
    bars = ax.bar(["Original AUC", "Occluded AUC"], vals_a,
                  color=colors_a, edgecolor="black", linewidth=0.7, width=0.5)
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    ax.set_title("(A) Fidelity\n(AUC drop = {:.4f})".format(
        results["fidelity_auc_drop"]), fontsize=10, fontweight="bold")
    ax.set_ylabel("AUC")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8)
    drop_color = "darkgreen" if results["fidelity_auc_drop"] > 0 else "darkred"
    ax.text(0.5, 0.03,
            "{} D = {:+.4f}".format(
                "+" if results["fidelity_auc_drop"] > 0 else "-",
                results["fidelity_auc_drop"]),
            ha="center", transform=ax.transAxes, fontsize=9, color=drop_color)

    # --- (B) Stability scatter ---
    ax = axes[1]
    sample_ids = np.arange(1, N + 1)
    ax.scatter(sample_ids, cos_sims, s=80, zorder=5,
               color="#59a14f", edgecolors="black", linewidth=0.7)
    ax.axhline(results["stability_mean_cos"], ls="--", color="#e15759",
               lw=1.2, label=f"Mean = {results['stability_mean_cos']:.3f}")
    ax.axhline(0.8, ls=":", color="gray", lw=0.8, alpha=0.6, label="Threshold 0.8")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(sample_ids)
    ax.set_xlabel("Sample #")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("(B) Stability\n(Gaussian noise perturbation)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5)

    # --- (C) Sparsity bars ---
    ax = axes[2]
    ax.bar([f"#{i+1}" for i in range(N)], gini_vals,
           color="#9c755f", edgecolor="black", linewidth=0.7)
    ax.axhline(results["sparsity_gini"], ls="--", color="#e15759",
               lw=1.2, label=f"Mean = {results['sparsity_gini']:.3f}")
    ax.axhline(0.5, ls=":", color="gray", lw=0.8, alpha=0.6,
               label="Sparse threshold 0.5")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("(C) Sparsity\n(Gini of |SHAP|)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5)

    # --- (D) Band localization horizontal bar ---
    ax = axes[3]
    bands  = list(results["band_localization"].keys())
    fracs  = [results["band_localization"][b] * 100 for b in bands]
    colors = [BAND_COLORS[b] for b in bands]
    hbars  = ax.barh(bands, fracs, color=colors, edgecolor="black", linewidth=0.7)
    ax.set_xlabel("Attribution Mass (%)")
    ax.set_title("(D) Band Localization\n(mean % of |SHAP| per band)",
                 fontsize=10, fontweight="bold")
    ax.bar_label(hbars, fmt="%.1f%%", fontsize=8, padding=3)
    ax.set_xlim(0, max(fracs) * 1.25 if fracs else 1)
    ax.invert_yaxis()

    plt.savefig("shap_quantitative_evaluation.png", dpi=150, bbox_inches="tight")
    #plt.show()


# ---------------------------------------------------------------------------
# 4.  Classification Metrics Report
# ---------------------------------------------------------------------------

def classify_and_report(model, test_batch, true_labels, num_classes, device="cuda",
                        class_names=None):
    """
    Runs the model on *test_batch* and reports:
      - Per-class Precision, Recall, F1-score
      - Macro-averaged ROC-AUC
      - Confusion matrix (as a figure)

    Parameters
    ----------
    model       : trained PyTorch model (eval mode assumed)
    test_batch  : torch.Tensor (N, C, H, W) on device
    true_labels : (N,) numpy int array of ground-truth class indices
    num_classes : int
    device      : 'cuda' or 'cpu'

    Returns
    -------
    dict with keys: precision, recall, f1, roc_auc, confusion_matrix
    """
    class_names = list(class_names or ["Alert", "Transition", "Drowsy"][:num_classes])
    if len(class_names) != num_classes:
        raise ValueError("class_names must have one name per model output class")

    model.eval()
    with torch.no_grad():
        outputs = model(test_batch.to(device))
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()   # (N, num_classes)
    pred_labels = np.argmax(probs, axis=1)

    # ------------------------------------------------------------------
    # Precision / Recall / F1
    # ------------------------------------------------------------------
    average = "binary" if num_classes == 2 else "macro"
    precision = precision_score(true_labels, pred_labels, average=average, zero_division=0)
    recall    = recall_score(true_labels, pred_labels,    average=average, zero_division=0)
    f1        = f1_score(true_labels, pred_labels,        average=average, zero_division=0)

    # ------------------------------------------------------------------
    # ROC-AUC
    # ------------------------------------------------------------------
    try:
        if num_classes == 2:
            roc_auc = roc_auc_score(true_labels, probs[:, 1])
        else:
            roc_auc = roc_auc_score(true_labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    # ------------------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------------------
    cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(num_classes))

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  [4] CLASSIFICATION METRICS REPORT")
    print("="*60)
    print(f"  Samples evaluated : {len(true_labels)}")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"  F1-Score          : {f1:.4f}")
    print(f"  ROC-AUC           : {roc_auc:.4f}")
    print("\n  Per-class report:")
    print(classification_report(true_labels, pred_labels,
                                labels=np.arange(num_classes),
                                target_names=class_names, zero_division=0))
    print("  Confusion Matrix:")
    print("  Predicted classes:", " | ".join(class_names))
    for r_idx, row in enumerate(cm):
        print(f"  True {class_names[r_idx]:12s}: {row.tolist()}")

    # ------------------------------------------------------------------
    # Figure: Confusion Matrix heatmap
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"wspace": 0.45})
    fig.suptitle("Classification Report", fontsize=13,
                 fontweight="bold", y=1.02)

    # Left: Confusion matrix heatmap
    ax_cm = axes[0]
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    tick_marks = np.arange(num_classes)
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels(class_names, fontsize=11)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels(class_names, fontsize=11, rotation=45, ha="right")
    ax_cm.set_xlabel("Predicted Label", fontsize=11)
    ax_cm.set_ylabel("True Label", fontsize=11)
    ax_cm.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    thresh = cm.max() / 2.0
    for row_i in range(cm.shape[0]):
        for col_j in range(cm.shape[1]):
            ax_cm.text(col_j, row_i, format(cm[row_i, col_j], "d"),
                       ha="center", va="center", fontsize=14, fontweight="bold",
                       color="white" if cm[row_i, col_j] > thresh else "black")

    # Right: Metric bar chart
    ax_bar = axes[1]
    metric_names  = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
    metric_values = [precision, recall, f1, roc_auc]
    bar_colors    = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    bars = ax_bar.bar(metric_names, metric_values, color=bar_colors,
                      edgecolor="black", linewidth=0.7, width=0.5)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.bar_label(bars, fmt="%.4f", fontsize=10, padding=3)
    ax_bar.axhline(0.5, ls="--", color="gray", lw=0.8, alpha=0.6, label="Chance (0.5)")
    ax_bar.set_ylabel("Score", fontsize=11)
    ax_bar.set_title("Classification Metrics", fontsize=12, fontweight="bold")
    ax_bar.legend(fontsize=9)
    ax_bar.tick_params(axis="x", labelsize=11)

    plt.savefig("classification_report.png", dpi=150, bbox_inches="tight")
    #plt.show()
    print("  Figure saved -> classification_report.png\n")

    return {
        "precision":       precision,
        "recall":          recall,
        "f1":              f1,
        "roc_auc":         roc_auc,
        "confusion_matrix": cm,
    }


def classify_loader_full(model, test_loader, device="cuda", class_names=None):
    """Evaluate every held-out window in batches; never report a SHAP subset as accuracy."""
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for images, batch_labels in test_loader:
            outputs = model(images.to(device))
            probabilities.append(torch.softmax(outputs, dim=1).cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    y_true = np.concatenate(labels)
    probs = np.concatenate(probabilities)
    y_pred = np.argmax(probs, axis=1)
    num_classes = probs.shape[1]
    class_names = list(class_names or ["Alert", "Transition", "Drowsy"][:num_classes])
    if len(class_names) != num_classes:
        raise ValueError("class_names must have one name per model output class")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    alert_recall = recall_score(y_true, y_pred, labels=[0], average=None,
                                zero_division=0)[0]
    try:
        roc_auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    print("\n" + "=" * 60)
    print("  HELD-OUT SUBJECT CLASSIFICATION REPORT (ALL WINDOWS)")
    print("=" * 60)
    print(f"  Samples evaluated : {len(y_true)}")
    print(f"  Accuracy          : {accuracy:.4f}")
    print(f"  Balanced accuracy : {balanced_accuracy:.4f}")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"  F1-Score          : {f1:.4f}")
    print(f"  ROC-AUC           : {roc_auc:.4f}")
    print(classification_report(y_true, y_pred,
                                labels=np.arange(num_classes),
                                target_names=class_names, zero_division=0))
    print("  Confusion Matrix:")
    print("  Predicted classes:", " | ".join(class_names))
    for r_idx, row in enumerate(cm):
        print(f"  True {class_names[r_idx]:12s}: {row.tolist()}")

    # ------------------------------------------------------------------
    # Figure: Confusion Matrix Heatmap & Metric Bar Chart
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.45})
    fig.suptitle("Held-Out Subject Classification Report (Full Test Set)", 
                 fontsize=13, fontweight="bold", y=1.02)
    # Left: Confusion matrix heatmap
    ax_cm = axes[0]
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels(class_names, fontsize=11)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels(class_names, fontsize=11, rotation=45, ha="right")
    ax_cm.set_xlabel("Predicted Label", fontsize=11)
    ax_cm.set_ylabel("True Label", fontsize=11)
    ax_cm.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    thresh = cm.max() / 2.0
    for row_i in range(cm.shape[0]):
        for col_j in range(cm.shape[1]):
            ax_cm.text(col_j, row_i, format(cm[row_i, col_j], "d"),
                       ha="center", va="center", fontsize=14, fontweight="bold",
                       color="white" if cm[row_i, col_j] > thresh else "black")
    # Right: Metric bar chart
    ax_bar = axes[1]
    metric_names = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
    metric_values = [precision, recall, f1, roc_auc]
    bar_colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    bars = ax_bar.bar(metric_names, metric_values, color=bar_colors,
                      edgecolor="black", linewidth=0.7, width=0.5)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.bar_label(bars, fmt="%.4f", fontsize=10, padding=3)
    ax_bar.axhline(0.5, ls="--", color="gray", lw=0.8, alpha=0.6, label="Chance (0.5)")
    ax_bar.set_ylabel("Score", fontsize=11)
    ax_bar.set_title("Classification Metrics", fontsize=12, fontweight="bold")
    ax_bar.legend(fontsize=9)
    ax_bar.tick_params(axis="x", labelsize=11)
    plt.savefig("classification_report.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure saved -> classification_report.png\n")

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "alert_recall": alert_recall,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_shap_analysis(model, background_loader, test_loader, device="cuda",
                      n_background=100, n_test=50, n_display=4,
                      top_k_frac=0.10, noise_std=0.05, n_stability_reps=5,
                      shap_nsamples=50, class_names=None,
                      quantitative_target_class=None):
    """
    End-to-end SHAP analysis pipeline.  Drop-in replacement for
    generate_shap_explanations() in Execution_Pipeline.py.

    Parameters
    ----------
    model              : trained VGG-16 (or any PyTorch model)
    background_loader  : DataLoader for background/baseline samples
    test_loader        : DataLoader for test samples to explain
    device             : 'cuda' or 'cpu'
    n_background       : number of background samples for GradientExplainer
    n_test             : stratified held-out samples for SHAP diagnostics
    n_display          : number of test samples shown in SHAP visual figures   (default 4)
    top_k_frac         : fraction of top-attribution pixels to occlude (fidelity)
    noise_std          : Gaussian noise sigma for stability test
    n_stability_reps   : number of noisy repetitions per sample (stability)
    shap_nsamples      : GradientExplainer Monte-Carlo samples per explanation
    """
    model.eval()
    class_names = tuple(class_names or ("Alert", "Transition", "Drowsy"))
    for param in model.parameters():
        param.requires_grad = True

    # ------------------------------------------------------------------
    # 1. Collect data batches
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    # Random background samples are less biased than the first shuffled batch.
    background_indices = rng.choice(len(background_loader.dataset),
                                   size=min(n_background, len(background_loader.dataset)),
                                   replace=False)
    bg_batch = torch.stack([background_loader.dataset[int(i)][0]
                            for i in background_indices]).to(device)

    # Stratified, reproducible explanation subset.  Classification is reported
    # separately for every held-out window below.
    test_labels = np.array([int(test_loader.dataset[i][1])
                            for i in range(len(test_loader.dataset))])
    chosen = []
    classes = np.unique(test_labels)
    per_class = max(1, n_test // len(classes))
    for label in classes:
        candidates = np.flatnonzero(test_labels == label)
        chosen.extend(rng.choice(candidates, size=min(per_class, len(candidates)), replace=False))
    remaining = min(n_test, len(test_labels)) - len(chosen)
    if remaining > 0:
        candidates = np.setdiff1d(np.arange(len(test_labels)), chosen)
        chosen.extend(rng.choice(candidates, size=remaining, replace=False))
    chosen = np.array(chosen, dtype=int)
    test_batch_all = torch.stack([test_loader.dataset[i][0] for i in chosen]).to(device)
    true_labels_all = test_labels[chosen]

    # Display subset (first n_display of the test set)
    n_display = min(n_display, len(chosen))
    test_batch_disp  = test_batch_all[:n_display]
    true_labels_disp = true_labels_all[:n_display]

    # ------------------------------------------------------------------
    # 2. Model predictions on FULL test set
    # ------------------------------------------------------------------
    with torch.no_grad():
        outputs_all     = model(test_batch_all)
        probs_all       = torch.softmax(outputs_all, dim=1).cpu().numpy()
        pred_labels_all = np.argmax(probs_all, axis=1)

    num_classes = outputs_all.shape[1]

    # Predictions on display subset
    probs_disp       = probs_all[:n_display]
    pred_labels_disp = pred_labels_all[:n_display]

    # ------------------------------------------------------------------
    # 3. SHAP values (computed on full test set for quantitative metrics)
    # ------------------------------------------------------------------
    print(f"[SHAP] Building GradientExplainer with {n_background} background samples...")
    explainer = shap.GradientExplainer(model, bg_batch)

    print(f"[SHAP] Computing SHAP values for {len(chosen)} stratified held-out samples...")
    # Fewer deterministic Monte-Carlo samples make development SHAP runs much
    # faster while preserving a fixed, comparable protocol.
    shap_values_all = explainer.shap_values(test_batch_all,
                                            nsamples=shap_nsamples, rseed=42)

    # SHAP values restricted to display subset
    if isinstance(shap_values_all, list):
        shap_values_disp = [sv[:n_display] for sv in shap_values_all]
    else:
        shap_values_disp = shap_values_all[:n_display]

    # Debug print
    print(f"[SHAP DEBUG] num_output_classes : {num_classes}")
    print(f"[SHAP DEBUG] type(shap_values)  : {type(shap_values_all)}")
    if isinstance(shap_values_all, np.ndarray):
        print(f"[SHAP DEBUG] shap_values.shape  : {shap_values_all.shape}")
    elif isinstance(shap_values_all, list):
        print(f"[SHAP DEBUG] len(shap_values)   : {len(shap_values_all)}")
        for idx, sv in enumerate(shap_values_all):
            arr = sv.cpu().detach().numpy() if hasattr(sv, "cpu") else np.array(sv)
            print(f"[SHAP DEBUG]   shap_values[{idx}].shape: {arr.shape}")

    # ------------------------------------------------------------------
    # 4. De-normalise ALL test images for plotting / stats
    # ------------------------------------------------------------------
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    test_images_all_np = test_batch_all.permute(0, 2, 3, 1).cpu().detach().numpy()
    test_images_all_np = np.clip(std * test_images_all_np + mean, 0, 1)

    # ====================================================================
    # Section 1 - SHAP Structure Verification
    #   stats  : all n_test samples  |  figure : first n_display samples
    # ====================================================================
    verify_shap_structure(shap_values_all, num_classes, test_images_all_np,
                          true_labels_all, pred_labels_all, probs_all,
                          n_display=n_display)

    # ====================================================================
    # Section 2 - Per-Band Attribution  (drowsy class)
    #   stats  : all n_test samples  |  figure : first n_display samples
    # ====================================================================
    plot_band_attribution(shap_values_all, num_classes, test_images_all_np,
                          true_labels_all, pred_labels_all,
                          class_idx=min(2, num_classes - 1), n_display=n_display)

    # ====================================================================
    # Section 3 - Quantitative Evaluation  (FULL test set)
    # ====================================================================
    metrics = evaluate_shap_quantitative(
        model, test_batch_all, shap_values_all, num_classes,
        true_labels_all, probs_all, background_batch=bg_batch, device=device,
        target_class=quantitative_target_class,
        top_k_frac=top_k_frac,
        noise_std=noise_std,
        n_stability_reps=n_stability_reps,
        shap_nsamples=shap_nsamples
    )

    # ====================================================================
    # Section 4 - Classification Metrics Report  (FULL test set)
    # ====================================================================
    clf_metrics = classify_loader_full(
        model, test_loader, device=device, class_names=class_names
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  Background samples        : {n_background}")
    print(f"  SHAP explanation samples  : {len(chosen)}")
    print(f"  Classification samples    : {len(test_loader.dataset)}")
    print(f"  Test samples (displayed)  : {n_display}")
    print(f"  Precision                 : {clf_metrics['precision']:.4f}")
    print(f"  Recall                    : {clf_metrics['recall']:.4f}")
    print(f"  F1-Score                  : {clf_metrics['f1']:.4f}")
    print(f"  ROC-AUC                   : {clf_metrics['roc_auc']:.4f}")
    print(f"  Fidelity    (AUC drop)    : {metrics['fidelity_auc_drop']:+.4f}")
    print(f"  Stability   (mean cos)    : {metrics['stability_mean_cos']:.4f}")
    print(f"  Sparsity    (mean Gini)   : {metrics['sparsity_gini']:.4f}")
    print("  Band Localization:")
    for band, frac in metrics["band_localization"].items():
        print(f"    {band:22s}: {frac*100:.1f}%")
    print("="*60)

    return {**metrics, **clf_metrics}
