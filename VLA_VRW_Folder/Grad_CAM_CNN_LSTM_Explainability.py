"""Grad-CAM for the CNN part of the CNN-LSTM EEG model.

This module is intentionally separate from ``Grad_CAM_Explainability.py``.
It produces one heatmap for each spectrogram in a temporal sequence and
explains one ordinal threshold at a time.
"""

import numpy as np
import matplotlib.pyplot as plt
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
            origin="lower", aspect="auto",
        )
        axes[time_index, 2].set_title("Overlay")
        for axis in axes[time_index]:
            axis.set_xlabel("Time")
            axis.set_ylabel("Frequency")

    target_name = "class >= Drowsy" if target_threshold == 1 else "class >= Transition"
    fig.suptitle(
        f"CNN-LSTM Grad-CAM: explaining {target_name}\n"
        f"True centre class: {true_class}; predicted class: {predicted_class}; "
        f"ordinal probabilities: {np.round(cumulative, 3)}"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return heatmaps, predicted_class, cumulative
