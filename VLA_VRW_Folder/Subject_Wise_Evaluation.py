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
from torch.utils.data import DataLoader, Subset

from PyTorch_Dataset_And_VGG16_Model_Setup import (
    DROZYSpectrogramDataset,
    build_small_cnn_classifier,
    transform_pipeline,
    train_vgg16,
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


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model, loader, device, prefix="Evaluation"):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            outputs = model(images.to(device))
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(batch_labels.numpy())

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
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

    # Subject-independent validation within lab only. Real remains untouched
    # until the final robustness/generalizability evaluation.
    lab_groups = np.asarray([f"lab_subject_{item['subject_id']}" for item in metadata_lab])
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    lab_dataset = DROZYSpectrogramDataset(X_lab, y_lab, transform=transform_pipeline)
    real_dataset = DROZYSpectrogramDataset(X_real, y_real, transform=transform_pipeline)
    real_loader = DataLoader(real_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fold_results_lab = []
    fold_results_real = []

    # --- Borrowed from Cross_Validate_Fz_Only.py ---
    for fold, (train_idx, val_idx) in enumerate(split.split(X_lab, y_lab, groups=lab_groups), start=1):
        set_reproducible_seed(SEED + fold)
        
        train_subjects = set(lab_groups[train_idx])
        val_subjects = set(lab_groups[val_idx])
        assert train_subjects.isdisjoint(val_subjects), "Subject leakage detected"

        print(f"\n{'='*30} FOLD {fold}/5 {'='*30}")
        print(f"Val subjects: {sorted(val_subjects)}")

        train_loader = DataLoader(Subset(lab_dataset, train_idx), batch_size=16, shuffle=True)
        val_loader = DataLoader(Subset(lab_dataset, val_idx), batch_size=16, shuffle=False)

        model = build_small_cnn_classifier(num_classes=NUM_CLASSES, dropout=0.35)
        trained_model = train_vgg16(
            model,
            train_loader,
            val_loader,
            epochs=25,
            lr=3e-4,
            device=device,
            early_stopping_patience=5,
            class_weight_power=1.0,
        )

        # 1. Evaluate on in-domain Lab validation subjects
        y_val_true, y_val_pred = evaluate_model(trained_model, val_loader, device, prefix=f"Fold {fold} Lab Val")
        fold_results_lab.append(balanced_accuracy_score(y_val_true, y_val_pred))

        # 2. Evaluate on out-of-domain Real dataset
        y_real_true, y_real_pred = evaluate_model(trained_model, real_loader, device, prefix=f"Fold {fold} Real Test")
        fold_results_real.append(balanced_accuracy_score(y_real_true, y_real_pred))

    # Summary statistics across all 5 folds:
    print("\n" + "=" * 60)
    print(f"Lab Val Balanced Acc:  {np.mean(fold_results_lab):.4f} +/- {np.std(fold_results_lab):.4f}")
    print(f"Real Held-out Balanced Acc: {np.mean(fold_results_real):.4f} +/- {np.std(fold_results_real):.4f}")

    if RUN_EXPLAINABILITY:
        from Grad_CAM_Explainability import show_gradcam_example
        from SHAP_Analysis_And_Evaluation import run_shap_analysis

        show_gradcam_example(
            trained_model,
            real_loader,
            device=device,
            class_names=CLASS_NAMES,
            target_class=2,
            sample_index=0,
            save_path="gradcam_real_drowsy.png",
        )
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
