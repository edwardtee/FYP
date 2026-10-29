import shap
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_eeg_shap_clean(test_images, shap_list, true_labels, pred_labels, probs=None, class_names=None):
    """
    Renders SHAP attributions with Ground Truth and Model Prediction labels.
    Guarantees sample-level indexing across (N, H, W, C) attribution tensors.
    """
    if class_names is None:
        class_names = ['Alert', 'Drowsy']
    else:
        class_names = list(class_names)
        
    num_samples = len(test_images)
    # Cap num_classes to the actual number of arrays in shap_list to prevent IndexError
    # when SHAP returns a different number of outputs than expected class count
    num_classes = len(shap_list)

    drowsy_shap = shap_list[1] if len(shap_list) > 1 else shap_list[0]

    # Extend class_names dynamically if SHAP outputs additional classes/channels
    while len(class_names) < num_classes:
        class_names.append(f"Class {len(class_names)}")
    
    fig, axes = plt.subplots(
        num_samples, 2, 
        figsize=(8.5, 2.6 * num_samples),
        gridspec_kw={'wspace': 0.35, 'hspace': 0.75}
    )
    
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    # Convert pixel Y-axis (0-224) to physical EEG frequency ticks
    freq_ticks = [0, 30, 60, 90, 224]
    freq_labels = ['0.5 Hz', '4 Hz', '8 Hz', '12 Hz', '30 Hz']
    
    # Convert pixel X-axis (0-224) to 5-second time ticks
    time_ticks = [0, 56, 112, 168, 224]
    time_labels = ['0s', '1.25s', '2.5s', '3.75s', '5s']

    for i in range(num_samples):
        # Format string labels for True and Predicted classes
        true_idx = int(true_labels[i])
        pred_idx = int(pred_labels[i])
        
        true_str = class_names[true_idx] if true_idx < len(class_names) else f"Class {true_idx}"
        pred_str = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
        
        # Green title if prediction matches ground truth, red if incorrect
        title_color = 'darkgreen' if true_idx == pred_idx else 'darkred'
        
        # 1. Input Spectrogram Rendering
        spec_2d = test_images[i]
        if spec_2d.ndim == 3:
            spec_2d = spec_2d.mean(axis=-1)
        
        im0 = axes[i, 0].imshow(spec_2d, cmap='viridis', aspect='auto', origin='lower')
        # Include Probability if available
        if probs is not None:
            pred_conf = probs[i, pred_idx] * 100
            title_text = f"Sample #{i+1}\nTrue: {true_str} | Pred: {pred_str} ({pred_conf:.1f}%)"
        else:
            title_text = f"Sample #{i+1}\nTrue: {true_str} | Pred: {pred_str}"
        axes[i, 0].set_title(title_text, fontsize=8.5, fontweight='bold', color=title_color)
        axes[i, 0].set_ylabel("EEG Frequency Band", fontsize=8)
        axes[i, 0].set_xlabel("Epoch Duration", fontsize=8)
        axes[i, 0].set_yticks(freq_ticks)
        axes[i, 0].set_yticklabels(freq_labels, fontsize=7)
        axes[i, 0].set_xticks(time_ticks)
        axes[i, 0].set_xticklabels(time_labels, fontsize=7)
        cbar0 = fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        cbar0.ax.tick_params(labelsize=7)

        # 2. Drowsiness Feature Attribution Map
        shap_attr = drowsy_shap[i]
        if shap_attr.ndim == 3:
            shap_attr = shap_attr.mean(axis=-1)

        vmax = np.percentile(np.abs(shap_attr), 99) + 1e-8
        
        im1 = axes[i, 1].imshow(shap_attr, cmap='seismic', aspect='auto', origin='lower', vmin=-vmax, vmax=vmax)
        axes[i, 1].set_title(f"Sample #{i+1}: Drowsiness (Class 1) Attribution", fontsize=9, fontweight='bold')
        axes[i, 1].set_xlabel("Epoch Duration", fontsize=8)
        axes[i, 1].set_yticks(freq_ticks)
        axes[i, 1].set_yticklabels(freq_labels, fontsize=7)
        axes[i, 1].set_xticks(time_ticks)
        axes[i, 1].set_xticklabels(time_labels, fontsize=7)
        
        cbar1 = fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=7)

    fig.suptitle("SHAP Explainability: EEG Spectrogram Feature Attribution", fontsize=11, fontweight='bold', y=0.98)
    plt.show()

def generate_shap_explanations(model, background_loader, test_loader, device='cuda'):
    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    # Baseline & Test Samples
    background_batch, _ = next(iter(background_loader))
    background_batch = background_batch[:10].to(device)

    test_batch, test_labels = next(iter(test_loader))
    test_batch = test_batch[:4].to(device)
    true_labels = test_labels[:4].cpu().numpy()

    # Model predictions
    # Temporarily disable Pytorch's automatic gradient calculation engine. 
    # As now only doing inference ( prediction ) and no more training. This can save up a lot of memory
    # and speed up the prediction process.
    with torch.no_grad():
        # test_batch: EEG spectrograms to be evaluated (4 samples, shape [4, 3, 224, 224]) representing [Batch size
        # Channels, Height, Width]
        # output: unnormalized prediction scores (logits) of shape [4, 2], for Alert vs. Drowsy
        outputs = model(test_batch)
        # torch.softmax(outputs,dim=1): Applies the softmax function to the output of the model along dimension 1.
        # This converts the raw logits into a probability distribution for each sample.
        # cpu().numpy(): Moves the tensor to the CPU (if it was on GPU) and converts it to a NumPy array.
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        pred_labels = np.argmax(probabilities, axis=1)

    # Compute SHAP Values
    # explainer: SHAP explainer object initialized with the VGG16 model and background data.
    # GradientExplainer: A SHAP explainer that computes feature attributions using gradients.
    # background_batch: Background data used to approximate the expected value of the output.
    explainer = shap.GradientExplainer(model, background_batch)
    # explainer.shap_values(test_batch): Computes SHAP attributions for the test data.
    shap_values = explainer.shap_values(test_batch)

    # --- DEBUG: print the raw SHAP output shape so we can see exactly what was returned ---
    num_output_classes = outputs.shape[1]
    print(f"[SHAP DEBUG] num_output_classes from model: {num_output_classes}")
    print(f"[SHAP DEBUG] type(shap_values): {type(shap_values)}")
    if isinstance(shap_values, np.ndarray):
        print(f"[SHAP DEBUG] shap_values.shape: {shap_values.shape}")
    elif isinstance(shap_values, list):
        print(f"[SHAP DEBUG] len(shap_values): {len(shap_values)}")
        for idx, sv in enumerate(shap_values):
            sv_arr = sv.cpu().detach().numpy() if hasattr(sv, 'cpu') else np.array(sv)
            print(f"[SHAP DEBUG]   shap_values[{idx}].shape: {sv_arr.shape}")

    # Convert output into a clean list of per-class 4D arrays: [(N, C, H, W), ...]
    # GradientExplainer returns one of:
    #   (a) ndarray of shape (num_classes, N, C, H, W)  -> split on axis 0
    #   (b) ndarray of shape (N, C, H, W)               -> single class, wrap in list
    #   (c) list of length num_classes, each (N, C, H, W)  -> use directly
    #   (d) list of length N (per-sample), each (C, H, W)  -> must stack into (num_classes, N, ...)
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 5:
            # GradientExplainer can return either:
            #   (num_classes, N, C, H, W)  -> split on axis 0
            #   (N, C, H, W, num_classes)  -> split on last axis  ← confirmed by debug
            if shap_values.shape[-1] == num_output_classes:
                # Last axis is classes: (N, C, H, W, num_classes)
                shap_list_raw = [shap_values[..., k] for k in range(num_output_classes)]
            elif shap_values.shape[0] == num_output_classes:
                # First axis is classes: (num_classes, N, C, H, W)
                shap_list_raw = [shap_values[k] for k in range(num_output_classes)]
            else:
                # Fallback: split on first axis and clip later
                shap_list_raw = [shap_values[k] for k in range(shap_values.shape[0])]
        elif shap_values.ndim == 4:
            # Shape (N, C, H, W) — single output
            shap_list_raw = [shap_values]
        else:
            shap_list_raw = [shap_values]
    elif isinstance(shap_values, list):
        # Convert all elements to numpy for shape inspection
        shap_values_np = []
        for sv in shap_values:
            shap_values_np.append(sv.cpu().detach().numpy() if hasattr(sv, 'cpu') else np.array(sv))

        first = shap_values_np[0]
        # Case (c): list length == num_output_classes and each element is (N, ...) batch array
        if len(shap_values_np) == num_output_classes and first.ndim >= 4:
            shap_list_raw = shap_values_np
        # Case (d): list length == N (batch size) — each element is a per-class array (num_classes, C, H, W)
        # or plain (C, H, W) needing stacking across samples
        else:
            # Stack across samples: result shape (N, num_classes, C, H, W) or (N, C, H, W)
            stacked = np.stack(shap_values_np, axis=0)  # (N, ...)
            if stacked.ndim == 5:
                # (N, num_classes, C, H, W) -> transpose to (num_classes, N, C, H, W)
                shap_list_raw = [stacked[:, k] for k in range(stacked.shape[1])]
            else:
                shap_list_raw = [stacked]
    else:
        shap_list_raw = [np.array(shap_values)]

    # Clip to model's true number of output classes before any further processing
    shap_list_raw = shap_list_raw[:num_output_classes]

    # Reorder dimensions for each class array from PyTorch (N, C, H, W) -> Matplotlib (N, H, W, C)
    shap_list_formatted = []
    for s in shap_list_raw:
        if hasattr(s, 'cpu'):
            s = s.cpu().detach().numpy()
        if s.ndim == 4 and s.shape[1] in [1, 3]:  # (N, C, H, W) channels-first
            s = np.transpose(s, (0, 2, 3, 1))      # -> (N, H, W, C)
        shap_list_formatted.append(s)

    print(f"[SHAP DEBUG] Final shap_list_formatted length: {len(shap_list_formatted)}")
    for idx, s in enumerate(shap_list_formatted):
        print(f"[SHAP DEBUG]   shap_list_formatted[{idx}].shape: {s.shape}")

    # De-normalize test images back to standard RGB range [0, 1]
    test_images_np = test_batch.permute(0, 2, 3, 1).cpu().detach().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    test_images_np = std * test_images_np + mean
    test_images_np = np.clip(test_images_np, 0, 1)

    # Render SHAP attributions with True and Predicted Labels
    plot_eeg_shap_clean(test_images_np, shap_list_formatted, true_labels, pred_labels, probabilities)

def generate_shap_explanations2(model, background_loader, test_loader, device='cuda'):
    """
    Computes SHAP values and triggers custom clean visualizer.
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    # Select background baseline and test samples
    background_batch, _ = next(iter(background_loader))
    background_batch = background_batch[:10].to(device)

    test_batch, _ = next(iter(test_loader))
    test_batch = test_batch[:4].to(device)

    # Compute raw SHAP attributions via GradientExplainer
    explainer = shap.GradientExplainer(model, background_batch)
    shap_values = explainer.shap_values(test_batch)

    # Unpack 5D/4D SHAP arrays into per-class list
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 5:
        shap_list = [shap_values[i] for i in range(shap_values.shape[0])]
    elif isinstance(shap_values, list):
        shap_list = shap_values
    else:
        shap_list = [shap_values]

    # Reorder PyTorch tensors (N, C, H, W) -> Matplotlib arrays (N, H, W, C)
    shap_list_formatted = [
        np.transpose(s, (0, 2, 3, 1)) if s.ndim == 4 and s.shape[1] in [1, 3] else s 
        for s in shap_list
    ]

    # De-normalize test background images back to standard [0, 1] range
    test_images_np = test_batch.permute(0, 2, 3, 1).cpu().detach().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    test_images_np = std * test_images_np + mean
    test_images_np = np.clip(test_images_np, 0, 1)

    # Render clean visualization figure with guarded class names list
    plot_eeg_shap_clean(test_images_np, shap_list_formatted)

def generate_shap_explanations1(model, background_loader, test_loader, device='cuda'):
    """
    Computes and visualizes SHAP heatmaps overlaid on EEG spectrograms.
    Preserves 4D tensor shapes (N, H, W, C) so SHAP and Matplotlib iterate over batches correctly.
    """
    model.eval()
    
    # Ensure gradients are enabled across all layers for SHAP computation
    for param in model.parameters():
        param.requires_grad = True

    # 1. Obtain background baseline (10 samples)
    background_batch, _ = next(iter(background_loader))
    background_batch = background_batch[:10].to(device)

    # 2. Select test samples to explain (4 samples)
    test_batch, test_labels = next(iter(test_loader))
    test_batch = test_batch[:4].to(device)

    # 3. Compute raw SHAP values via GradientExplainer
    explainer = shap.GradientExplainer(model, background_batch)
    shap_values = explainer.shap_values(test_batch)

    # 4. Convert shap_values into a list of per-class arrays
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 5:
            shap_list = [shap_values[i] for i in range(shap_values.shape[0])]
        else:
            shap_list = [shap_values]
    elif isinstance(shap_values, list):
        shap_list = shap_values
    else:
        shap_list = [np.array(shap_values)]

    # 5. Process SHAP values while strictly preserving 4D shape (N, H, W, C)
    processed_shap = []
    for s in shap_list:
        if hasattr(s, 'cpu'):
            s = s.cpu().detach().numpy()
            
        # Reorder channels-first (N, C, H, W) -> channels-last (N, H, W, C)
        if s.ndim == 4 and s.shape[1] in [1, 3]:
            s = np.transpose(s, (0, 2, 3, 1))
            
        # If 4D with 3 RGB channels, sum across channels while KEEPING channel axis -> (N, H, W, 1)
        if s.ndim == 4 and s.shape[-1] == 3:
            s_sum = s.sum(axis=-1, keepdims=True)
            processed_shap.append(s_sum)
        elif s.ndim == 4:
            processed_shap.append(s)
        elif s.ndim == 3:
            # If 3D (N, H, W), restore 4D with single channel dim -> (N, H, W, 1)
            processed_shap.append(np.expand_dims(s, axis=-1))
        else:
            processed_shap.append(s)

    # 6. Format background test images for Matplotlib: (N, C, H, W) -> (N, H, W, C)
    test_images_np = test_batch.permute(0, 2, 3, 1).cpu().detach().numpy()
    
    # Reverse ImageNet normalization to make spectrogram features clearly visible
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    test_images_np = std * test_images_np + mean
    test_images_np = np.clip(test_images_np, 0, 1)

    # 7. Render SHAP attribution plot
    plt.figure(figsize=(12, 8))
    shap.image_plot(processed_shap, test_images_np, show=False)
    plt.suptitle("SHAP Explainability: Spectrogram Feature Attribution\n(Left: Input Spectrogram | Center: Alert Class [0] | Right: Drowsy Class [1])", fontsize=11)
    plt.tight_layout()
    plt.show()