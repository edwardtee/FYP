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
    build_small_cnn_ordinal_classifier,
    ordinal_logits_to_classes,
    transform_pipeline,
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
USE_ORDINAL_MODEL = True


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model, loader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            outputs = model(images.to(device))
            if USE_ORDINAL_MODEL:
                batch_predictions = ordinal_logits_to_classes(outputs)
            else:
                batch_predictions = outputs.argmax(dim=1)
            predictions.extend(batch_predictions.cpu().numpy())
            labels.extend(batch_labels.numpy())

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    print("\nHeld-out real-condition evaluation")
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

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    real_loader = DataLoader(real_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_small_cnn_ordinal_classifier(num_classes=NUM_CLASSES, dropout=0.35)
    trained_model = train_ordinal_model(
        model,
        train_loader,
        val_loader,
        num_classes=NUM_CLASSES,
        epochs=25,
        lr=3e-4,
        device=device,
        early_stopping_patience=5,
    )

    # This is the only evaluation on the real condition.
    evaluate_model(trained_model, real_loader, device)

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
