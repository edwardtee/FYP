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

