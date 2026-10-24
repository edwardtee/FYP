"""Train on VLA_VRW lab data and evaluate on held-out VLA_VRW real data.

The CSV export must be generated first with Export_VLA_VRW_To_CSV.py.
This benchmark deliberately does not mix lab and real recordings during
training or validation: lab is development/training data and real is the
final robustness/generalizability evaluation set.
"""

import random

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from PyTorch_Dataset_And_VGG16_Model_Setup import (
    DROZYSpectrogramDataset,
    build_small_cnn_classifier,
    build_small_cnn_ordinal_classifier,
    ordinal_logits_to_classes,
    transform_pipeline,
    train_vgg16,
    train_ordinal_model,
)

from VLA_VRW_CSV_Dataset import load_benchmark_split

SEED = 42
NUM_CLASSES = 3
CLASS_NAMES = ("Alert", "Transition", "Drowsy")
# Keep the stronger Fz-only baseline for the benchmark. The three-channel
# experiment reduced held-out balanced accuracy in the current split.
MODEL_CHANNELS = ("Fz_Pz_uv",)
CSV_DATASET_PATH = r"D:\FYP\VLA_VRW_CSV"
RUN_EXPLAINABILITY = False
# First controlled improvement experiment: ordinary three-class CNN with
# inverse-frequency weighted cross-entropy.  The ordinal model remains below
# for comparison by changing this flag and the training block.
USE_ORDINAL_MODEL = True
SMOOTHING_CANDIDATES = (1, 3, 5)


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _model_scores(outputs):
    """Return scores that preserve the model's original decision rule."""
    if not USE_ORDINAL_MODEL:
        return torch.softmax(outputs, dim=1)

    # Ordinal outputs are cumulative probabilities:
    # p1 = P(class >= Transition), p2 = P(class >= Drowsy).
    return torch.sigmoid(outputs)


def _smooth_scores(scores, metadata, window_size=3):
    """Average consecutive windows within each subject only.

    Samples from different subjects or gaps in window_index are never mixed.
    A centered window is used, with a shorter window at recording boundaries.
    """
    if window_size <= 1:
        return scores
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    smoothed = np.empty_like(scores)
    half = window_size // 2
    groups = {}
    for index, item in enumerate(metadata):
        key = (item["condition"], item["subject_id"])
        groups.setdefault(key, []).append(index)

    for indices in groups.values():
        indices.sort(key=lambda index: metadata[index]["window_index"])
        for position, index in enumerate(indices):
            selected = []
            center_window = metadata[index]["window_index"]
            for candidate_position in range(
                max(0, position - half), min(len(indices), position + half + 1)
            ):
                candidate = indices[candidate_position]
                candidate_window = metadata[candidate]["window_index"]
                if abs(candidate_window - center_window) <= half:
                    selected.append(candidate)
            smoothed[index] = scores[selected].mean(axis=0)
    return smoothed


def evaluate_model(model, loader, metadata, device, prefix, smooth_windows=1):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            outputs = model(images.to(device))
            scores.extend(_model_scores(outputs).cpu().numpy())
            labels.extend(batch_labels.numpy())

    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(metadata) != len(scores):
        raise ValueError("metadata length must match evaluation samples")
    scores = _smooth_scores(
        scores, metadata, window_size=smooth_windows
    )
    if USE_ORDINAL_MODEL:
        predictions = (scores >= 0.5).sum(axis=1).astype(int)
    else:
        predictions = scores.argmax(axis=1)
    print(f"\n{prefix}")
    print(classification_report(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))
    print("Balanced accuracy:", f"{balanced_accuracy_score(labels, predictions):.4f}")
    print("Confusion matrix [rows=true, columns=predicted]:")
    print(confusion_matrix(labels, predictions, labels=np.arange(NUM_CLASSES)))
    return labels, predictions


if __name__ == "__main__":
    set_reproducible_seed(SEED)

    (X_lab, y_lab, perclos_lab, metadata_lab), (X_real, y_real, perclos_real, metadata_real) = (
        load_benchmark_split(CSV_DATASET_PATH, channels=MODEL_CHANNELS)
    )

    print(f"Model EEG channels: {MODEL_CHANNELS}")
    print(f"Lab samples: {X_lab.shape}; labels: {y_lab.shape}")
    print(f"Real samples: {X_real.shape}; labels: {y_real.shape}")
    print(f"PERCLOS classes: {CLASS_NAMES}")
    print("Lab class counts:", np.bincount(y_lab, minlength=NUM_CLASSES).tolist())
    print("Real class counts:", np.bincount(y_real, minlength=NUM_CLASSES).tolist())

    # Subject-independent validation within lab only. Real remains untouched
    # until the final robustness/generalizability evaluation.
    lab_groups = np.asarray([
        f"lab_subject_{item['subject_id']}" for item in metadata_lab
    ])
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(split.split(X_lab, y_lab, groups=lab_groups))
    train_subjects = set(lab_groups[train_idx])
    val_subjects = set(lab_groups[val_idx])
    assert train_subjects.isdisjoint(val_subjects), "Subject leakage detected"
    print(f"Lab train subjects: {sorted(train_subjects)}")
    print(f"Lab validation subjects: {sorted(val_subjects)}")

    lab_dataset = DROZYSpectrogramDataset(X_lab, y_lab, transform=transform_pipeline)
    real_dataset = DROZYSpectrogramDataset(X_real, y_real, transform=transform_pipeline)
    train_set = torch.utils.data.Subset(lab_dataset, train_idx)
    val_set = torch.utils.data.Subset(lab_dataset, val_idx)
    val_metadata = [metadata_lab[index] for index in val_idx]

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    real_loader = DataLoader(real_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if USE_ORDINAL_MODEL:
        model = build_small_cnn_ordinal_classifier(num_classes=NUM_CLASSES, dropout=0.35)
        trained_model = train_ordinal_model(
            model, train_loader, val_loader, num_classes=NUM_CLASSES,
            epochs=25, lr=3e-4, device=device, early_stopping_patience=5,
        )
    else:
        model = build_small_cnn_classifier(num_classes=NUM_CLASSES, dropout=0.35)
        trained_model = train_vgg16(
            model, train_loader, val_loader, epochs=25, lr=3e-4,
            device=device, early_stopping_patience=5, class_weight_power=1.0,
        )

    # Select the smoothing length using lab validation data only. The real set
    # remains completely held out until this choice has been made.
    print("\nSelecting temporal smoothing length on lab validation data")
    validation_scores = {}
    for smoothing_windows in SMOOTHING_CANDIDATES:
        val_labels, val_predictions = evaluate_model(
            trained_model, val_loader, val_metadata, device,
            prefix=f"Lab validation ({smoothing_windows}-window smoothing)",
            smooth_windows=smoothing_windows,
        )
        validation_scores[smoothing_windows] = balanced_accuracy_score(
            val_labels, val_predictions
        )

    selected_smoothing_windows = max(
        validation_scores,
        key=lambda windows: (validation_scores[windows], -windows),
    )
    print("Validation smoothing scores:", {
        windows: round(score, 4)
        for windows, score in validation_scores.items()
    })
    print(f"Selected smoothing length: {selected_smoothing_windows} windows")

    # Report the unsmoothed baseline and then evaluate the real set once using
    # the smoothing length selected without looking at real labels.
    evaluate_model(
        trained_model, real_loader, metadata_real, device,
        prefix="Held-out real-condition evaluation (single windows)",
        smooth_windows=1,
    )
    evaluate_model(
        trained_model, real_loader, metadata_real, device,
        prefix=("Held-out real-condition evaluation "
                f"({selected_smoothing_windows}-window smoothing)"),
        smooth_windows=selected_smoothing_windows,
    )

    if RUN_EXPLAINABILITY:
        from Grad_CAM_Explainability import show_gradcam_example

        show_gradcam_example(
            trained_model,
            real_loader,
            device=device,
            class_names=("PERCLOS >= 0.35", "PERCLOS > 0.70"),
            target_class=1,
            sample_index=0,
            save_path="gradcam_real_drowsy.png",
        )
        if USE_ORDINAL_MODEL:
            print("SHAP is disabled for ordinal output: explain one ordinal "
                  "threshold at a time before using multiclass SHAP summaries.")
        else:
            from SHAP_Analysis_And_Evaluation import run_shap_analysis
            run_shap_analysis(
                trained_model,
                train_loader,
                real_loader,
                device=device,
                n_background=30,
                n_test=20,
                n_display=4,
                n_stability_reps=2,
                shap_nsamples=50,
                class_names=CLASS_NAMES,
                quantitative_target_class=2,
            )
