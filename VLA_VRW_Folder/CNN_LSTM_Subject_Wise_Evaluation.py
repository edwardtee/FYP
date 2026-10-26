"""Five-fold subject-independent evaluation of a CNN-LSTM ordinal model.

Each sample is a sequence of five consecutive 8-second spectrograms (40 s).
The target is the label of the centre window.  Lab subjects are used for
training/validation and the real condition remains held out for testing.
"""

import random
import os
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Subset

from PyTorch_Dataset_And_VGG16_Model_Setup import (
    DROZYSpectrogramDataset,
    build_cnn_lstm_ordinal_classifier,
    transform_pipeline,
    train_ordinal_model,
)
from VLA_VRW_CSV_Dataset import load_benchmark_split
from Grad_CAM_CNN_LSTM_Explainability import (
    show_cnn_lstm_gradcam_example,
)


SEED = 42
NUM_CLASSES = 3
SEQUENCE_LENGTH = 5
CLASS_NAMES = ("Alert", "Transition", "Drowsy")
MODEL_CHANNELS = ("Fz_Pz_uv",)
CSV_DATASET_PATH = r"D:\FYP\VLA_VRW_CSV"
RUN_TRAINING = False  # Set to False once you have trained, or when you just want Grad-CAM


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_sequence_indices(metadata, candidate_indices, sequence_length):
    """Create global index sequences without crossing subjects or gaps."""
    half = sequence_length // 2
    groups = {}
    for index in candidate_indices:
        item = metadata[int(index)]
        key = (item["condition"], item["subject_id"])
        groups.setdefault(key, []).append(int(index))

    sequences = []
    centres = []
    for indices in groups.values():
        indices.sort(key=lambda index: metadata[index]["window_index"])
        for position in range(half, len(indices) - half):
            window_indices = indices[position - half:position + half + 1]
            window_numbers = [metadata[index]["window_index"] for index in window_indices]
            if window_numbers != list(range(window_numbers[0], window_numbers[-1] + 1)):
                continue
            sequences.append(window_indices)
            centres.append(indices[position])
    return sequences, centres


class SpectrogramSequenceDataset(Dataset):
    def __init__(self, frame_dataset, sequences, labels):
        self.frame_dataset = frame_dataset
        self.sequences = sequences
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        frames = [self.frame_dataset[frame_index][0] for frame_index in self.sequences[index]]
        return torch.stack(frames, dim=0), torch.tensor(self.labels[index], dtype=torch.long)


def evaluate_sequence_model(model, loader, device, prefix):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for sequences, batch_labels in loader:
            scores.append(torch.sigmoid(model(sequences.to(device))).cpu().numpy())
            labels.append(batch_labels.numpy())

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    predictions = (scores >= 0.5).sum(axis=1).astype(int)
    print(f"\n{prefix}")
    print(classification_report(
        labels, predictions, labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES, zero_division=0,
    ))
    print("Balanced accuracy:", f"{balanced_accuracy_score(labels, predictions):.4f}")
    print("Confusion matrix [rows=true, columns=predicted]:")
    print(confusion_matrix(labels, predictions, labels=np.arange(NUM_CLASSES)))
    return labels, predictions


if __name__ == "__main__":
    set_reproducible_seed(SEED)
    (X_lab, y_lab, _, metadata_lab), (X_real, y_real, _, metadata_real) = (
        load_benchmark_split(CSV_DATASET_PATH, channels=MODEL_CHANNELS)
    )

    lab_groups = np.asarray([
        f"lab_subject_{item['subject_id']}" for item in metadata_lab
    ])
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    lab_frame_dataset = DROZYSpectrogramDataset(X_lab, y_lab, transform=transform_pipeline)
    real_frame_dataset = DROZYSpectrogramDataset(X_real, y_real, transform=transform_pipeline)
    real_sequences, real_centres = make_sequence_indices(
        metadata_real, range(len(X_real)), SEQUENCE_LENGTH
    )
    real_sequence_dataset = SpectrogramSequenceDataset(
        real_frame_dataset,
        real_sequences,
        y_real[real_centres],
    )
    real_loader = DataLoader(real_sequence_dataset, batch_size=4, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_results_lab, fold_results_real = [], []

    if RUN_TRAINING:
        for fold, (train_idx, val_idx) in enumerate(
            split.split(X_lab, y_lab, groups=lab_groups), start=1
        ):
            set_reproducible_seed(SEED + fold)
            train_subjects = set(lab_groups[train_idx])
            val_subjects = set(lab_groups[val_idx])
            assert train_subjects.isdisjoint(val_subjects)

            train_sequences, train_centres = make_sequence_indices(
                metadata_lab, train_idx, SEQUENCE_LENGTH
            )
            val_sequences, val_centres = make_sequence_indices(
                metadata_lab, val_idx, SEQUENCE_LENGTH
            )
            train_sequence_dataset = SpectrogramSequenceDataset(
                lab_frame_dataset, train_sequences, y_lab[train_centres]
            )
            val_sequence_dataset = SpectrogramSequenceDataset(
                lab_frame_dataset, val_sequences, y_lab[val_centres]
            )
            train_loader = DataLoader(train_sequence_dataset, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_sequence_dataset, batch_size=4, shuffle=False)

            print(f"\n{'=' * 30} FOLD {fold}/5 {'=' * 30}")
            print(f"Validation subjects: {sorted(val_subjects)}")
            print(f"Train sequences: {len(train_sequence_dataset)}; "
                f"validation sequences: {len(val_sequence_dataset)}")

            model = build_cnn_lstm_ordinal_classifier(
                num_classes=NUM_CLASSES, hidden_size=128, dropout=0.35
            )
            model = train_ordinal_model(
                model, train_loader, val_loader, num_classes=NUM_CLASSES,
                epochs=25, lr=3e-4, device=device, early_stopping_patience=5,
            )
            # Save model after fold training
            torch.save(model.state_dict(), f"cnn_lstm_fold{fold}.pt")

            y_val, p_val = evaluate_sequence_model(
                model, val_loader, device, f"Fold {fold} Lab validation"
            )
            y_real, p_real = evaluate_sequence_model(
                model, real_loader, device, f"Fold {fold} Real test"
            )
            fold_results_lab.append(balanced_accuracy_score(y_val, p_val))
            fold_results_real.append(balanced_accuracy_score(y_real, p_real))

        print("\n" + "=" * 60)
        print(f"CNN-LSTM Lab Val Balanced Acc: {np.mean(fold_results_lab):.4f} "
            f"+/- {np.std(fold_results_lab):.4f}")
        print(f"CNN-LSTM Real Balanced Acc: {np.mean(fold_results_real):.4f} "
            f"+/- {np.std(fold_results_real):.4f}")

    # --- Explainability (Loads saved weights instantly) ---
    checkpoint_path = "VLA_VRW_Folder/pt_file/cnn_lstm_fold1.pt"
    if os.path.exists(checkpoint_path):
        print(f"\nLoading {checkpoint_path} for Grad-CAM...")
        model = build_cnn_lstm_ordinal_classifier(num_classes=NUM_CLASSES, hidden_size=128, dropout=0.35)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        heatmaps, predicted_class, cumulative_probabilities = show_cnn_lstm_gradcam_example(
            model,
            real_loader,
            device=device,
            target_threshold=1,         # 0 = Transition, 1 = Drowsy
            sequence_index=1,
            save_path="cnn_lstm_gradcam_f1s1_drowsy.png",
        )
        print("Grad-CAM saved to cnn_lstm_gradcam_f1s1_drowsy.png")
        show_cnn_lstm_gradcam_example(
            model,
            real_loader,
            device=device,
            target_threshold=0,         # 0 = Transition, 1 = Drowsy
            sequence_index=1,
            save_path="cnn_lstm_gradcam_f1s1_transition.png",
        )
        print("Grad-CAM saved to cnn_lstm_gradcam_f1s1_transition.png")
    else:
        print(f"Checkpoint {checkpoint_path} not found. Please run training first.")
