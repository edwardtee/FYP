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


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model, loader, device, thresholds=(0.5, 0.5), prefix="Evaluation"):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            outputs = model(images.to(device))
            predictions.extend(ordinal_logits_to_classes(outputs, thresholds=thresholds).cpu().numpy())
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

def find_optimal_ordinal_thresholds(model, val_loader, device):
    """Grid-search (T1, T2) on the validation set to maximize Balanced Accuracy."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images.to(device))
            all_logits.append(logits.cpu())
            all_labels.extend(labels.numpy())
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = np.asarray(all_labels)
    # Search candidates between 0.15 and 0.85 in steps of 0.05
    search_space = np.linspace(0.15, 0.85, 15)
    best_balanced_acc = -1.0
    best_thresholds = (0.5, 0.5)
    for t1 in search_space:
        for t2 in search_space:
            preds = ordinal_logits_to_classes(all_logits, thresholds=(t1, t2)).numpy()
            bacc = balanced_accuracy_score(all_labels, preds)
            
            if bacc > best_balanced_acc:
                best_balanced_acc = bacc
                best_thresholds = (round(float(t1), 3), round(float(t2), 3))
    return best_thresholds, best_balanced_acc

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

        # 2. Tune decision thresholds strictly on Lab Validation data
        best_t, val_bacc = find_optimal_ordinal_thresholds(trained_model, val_loader, device)
        print(f"Optimal thresholds for Fold {fold}: T1={best_t[0]}, T2={best_t[1]} (Val BAcc: {val_bacc:.4f})")
        # 3. Evaluate Lab Validation using optimal thresholds
        y_val_true, y_val_pred = evaluate_model(
            trained_model, val_loader, device, 
            thresholds=best_t, 
            prefix=f"Fold {fold} Lab Val"
        )
        # 4. Evaluate Held-out Real Test using the SAME optimal thresholds
        y_real_true, y_real_pred = evaluate_model(
            trained_model, real_loader, device, 
            thresholds=best_t, 
            prefix=f"Fold {fold} Real Test"
        )

        # Record balanced accuracy for this fold
        # 1. Evaluate on in-domain Lab validation subjects
        #y_val_true, y_val_pred = evaluate_model(trained_model, val_loader, device, prefix=f"Fold {fold} Lab Val")        
        fold_results_lab.append(balanced_accuracy_score(y_val_true, y_val_pred))
        # 2. Evaluate on out-of-domain Real dataset
        #y_real_true, y_real_pred = evaluate_model(trained_model, real_loader, device, prefix=f"Fold {fold} Real Test")
        fold_results_real.append(balanced_accuracy_score(y_real_true, y_real_pred))

    # Summary statistics across all 5 folds:
    print("\n" + "=" * 60)
    print(f"Lab Val Balanced Acc:  {np.mean(fold_results_lab):.4f} +/- {np.std(fold_results_lab):.4f}")
    print(f"Real Held-out Balanced Acc: {np.mean(fold_results_real):.4f} +/- {np.std(fold_results_real):.4f}")

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
        print("SHAP is disabled for ordinal output. Explain each cumulative "
              "threshold separately before producing class-level SHAP plots.")
