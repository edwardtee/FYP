"""Nested subject-level cross-validation for the final Fz-only model.

Outer folds are used only for final testing.  An inner subject-level split of
the remaining data controls early stopping, so no outer test subject influences
model selection.  SHAP is intentionally excluded: run it once afterwards on
the selected final Fz-only model, not five times during performance evaluation.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold

from STFT_Spectrogram_Generation import load_drozy_dataset
from PyTorch_Dataset_And_VGG16_Model_Setup import (
    DROZYSpectrogramDataset, build_small_cnn_classifier, build_vgg16_classifier, transform_pipeline, train_vgg16,
)
from SHAP_Analysis_And_Evaluation import classify_loader_full


DATASET_ROOT = r"D:\FYP\DROZY\DROZY"
OUTPUT_CSV = Path(r"D:\FYP\Drozy F\fz_only_nested_cv_results_CNN.csv")
SUMMARY_CSV = Path(r"D:\FYP\Drozy F\fz_only_nested_cv_mean_sd_CNN.csv")
SEED = 42
N_OUTER_FOLDS = 5
MODEL_CHANNELS = ("Fz",)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_data, y_data, subject_groups = load_drozy_dataset(
        DATASET_ROOT, return_groups=True, eeg_channels=MODEL_CHANNELS
    )
    dataset = DROZYSpectrogramDataset(X_data, y_data, transform=transform_pipeline)
    outer_cv = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                                    random_state=SEED)
    fold_results = []

    for fold, (outer_train_idx, outer_test_idx) in enumerate(
            outer_cv.split(X_data, y_data, groups=subject_groups), start=1):
        # Validation subjects are selected strictly within the outer training set.
        inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True,
                                        random_state=SEED + fold)
        inner_train_relative, inner_val_relative = next(inner_cv.split(
            X_data[outer_train_idx], y_data[outer_train_idx],
            groups=subject_groups[outer_train_idx]
        ))
        train_idx = outer_train_idx[inner_train_relative]
        val_idx = outer_train_idx[inner_val_relative]
        outer_test_subjects = sorted(set(subject_groups[outer_test_idx]))
        assert set(subject_groups[train_idx]).isdisjoint(outer_test_subjects)
        assert set(subject_groups[val_idx]).isdisjoint(outer_test_subjects)

        print("\n" + "=" * 72)
        print(f"OUTER FOLD {fold}/{N_OUTER_FOLDS} | Test subjects: {outer_test_subjects}")
        print("=" * 72)
        set_seed(SEED + fold)
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16,
                                  shuffle=True, generator=torch.Generator().manual_seed(SEED + fold))
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16, shuffle=False)
        test_loader = DataLoader(Subset(dataset, outer_test_idx), batch_size=16, shuffle=False)

        #model = build_vgg16_classifier(num_classes=2, freeze_features=True)
        #model = train_vgg16(model, train_loader, val_loader, epochs=8, device=device)
        model = build_small_cnn_classifier(num_classes=2, dropout=0.35)
        model = train_vgg16(
            model,
            train_loader,
            val_loader,
            epochs=15,
            lr=1e-3,
            device=device
        )
        metrics = classify_loader_full(model, test_loader, device=device)
        metrics["fold"] = fold
        metrics["test_subjects"] = ",".join(map(str, outer_test_subjects))
        metrics["n_test_windows"] = len(outer_test_idx)
        fold_results.append(metrics)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(fold_results)
    metric_columns = ["accuracy", "balanced_accuracy", "alert_recall",
                      "precision", "recall", "f1", "roc_auc"]
    summary = pd.DataFrame({
        "metric": metric_columns,
        "mean": [results[col].mean() for col in metric_columns],
        "std": [results[col].std(ddof=1) for col in metric_columns],
    })
    results.to_csv(OUTPUT_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 72)
    print("FZ-ONLY NESTED SUBJECT-LEVEL CROSS-VALIDATION: MEAN ± SD")
    print("=" * 72)
    for _, row in summary.iterrows():
        print(f"  {row['metric']:18s}: {row['mean']:.4f} ± {row['std']:.4f}")
    print(f"\nSaved fold results: {OUTPUT_CSV}")
    print(f"Saved mean ± SD summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
