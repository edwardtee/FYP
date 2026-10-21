import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from Explainable_AI_Integration_using_SHAP import generate_shap_explanations
from PyTorch_Dataset_And_VGG16_Model_Setup import DROZYSpectrogramDataset, build_vgg16_classifier, transform_pipeline, train_vgg16
from STFT_Spectrogram_Generation import load_drozy_dataset
from SHAP_Analysis_And_Evaluation import run_shap_analysis
from Grad_CAM_Explainability import show_gradcam_example

SEED = 42
MODEL_CHANNELS = ("Fz",)  # RGB order for the multichannel comparison


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_reproducible_seed(SEED)
    # Path where you extracted the DROZY database
    DROZY_DATASET_PATH = r"d:\FYP\DROZY\DROZY"

    # Load real spectrogram images and binary labels from DROZY PSG and KSS files
    X_data, y_data, subject_groups = load_drozy_dataset(
        DROZY_DATASET_PATH, window_sec=5, overlap_sec=2.5,
        return_groups=True, eeg_channels=MODEL_CHANNELS
    )

    print(f"Model input channels (R, G, B): {MODEL_CHANNELS}")
    print(f"Loaded Spectrograms Shape: {X_data.shape}") # Expect (N, 224, 224, 3)
    print(f"Loaded Labels Shape: {y_data.shape}")             # Expect (N,)

    # Wrap inside dataset loader
    dataset = DROZYSpectrogramDataset(X_data, y_data, transform=transform_pipeline)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Subject-independent development fold.  Do not use random window-level
    # splitting: overlapping windows from one recording would otherwise leak.
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(split.split(X_data, y_data, groups=subject_groups))
    train_subjects, val_subjects = set(subject_groups[train_idx]), set(subject_groups[val_idx])
    assert train_subjects.isdisjoint(val_subjects), "Subject leakage detected"
    print(f"Train subjects: {sorted(train_subjects)}")
    print(f"Held-out subjects: {sorted(val_subjects)}")
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    # 1. Build Model
    vgg16_model = build_vgg16_classifier(num_classes=2, freeze_features=True)

    # 2. Train Model
    trained_model = train_vgg16(vgg16_model, train_loader, val_loader, epochs=8, device=device)


    show_gradcam_example(
        trained_model,
        val_loader,
        device=device,
        target_class=None,
        sample_index=3,
        save_path="Grad_Cam_Example_n3_nperseg512.png"
    )

    show_gradcam_example(
        trained_model,
        val_loader,
        device=device,
        target_class=None,
        sample_index=4,
        save_path="Grad_Cam_Example_n4_nperseg512.png"
    )

    show_gradcam_example(
        trained_model,
        val_loader,
        device=device,
        target_class=None,
        sample_index=5,
        save_path="Grad_Cam_Example_n5_nperseg512.png"
    )

    # 3. Generate SHAP Visualizations
    #generate_shap_explanations(trained_model, train_loader, val_loader, device=device)
    # Fast, reproducible SHAP run for development.  Increase these only for
    # the final selected model after cross-validation is complete.
    '''run_shap_analysis(trained_model, train_loader, val_loader, device=device,
                      n_background=30, n_test=20, n_display=4,
                      n_stability_reps=2, shap_nsamples=50)'''
