import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms

# Preprocessing transforms matching standard ImageNet normalization
transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SpectrogramDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)

def build_vgg16_classifier(num_classes=2, freeze_features=True):
    """
    Instantiates VGG-16 model with optional feature freezing for fine-tuning.
    """
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    
    if freeze_features:
        # Freeze lower convolutional layers to preserve low-level features
        for param in model.features[:24].parameters():
            param.requires_grad = False
            
        # Keep higher conv layers (24-30) trainable for SHAP gradient propagation & fine-tuning
        for param in model.features[24:].parameters():
            param.requires_grad = True
    else:
        # Unfreeze all parameters across the network
        for param in model.parameters():
            param.requires_grad = True
            
    # Modify final fully connected output layer for binary classification (Alert vs Drowsy)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    
    return model


class SmallEEGCNN(nn.Module):
    """Compact CNN for 224x224 single-channel Fz spectrograms replicated to RGB.

    Adaptive pooling makes the classifier less dependent on the exact spatial
    size of the spectrogram and keeps the parameter count much lower than VGG16.
    """

    def __init__(self, num_classes=2, dropout=0.35):
        super().__init__()
        ''' Input (N, 3, 224, 224) -> Output (N, num_classes) 
        Define a small CNN architecture with convolutional layers, batch normalization, ReLU activations, and max pooling.
        Expanse the number of channels in each layer to capture more complex features while keeping the model compact. 
        (3 input channels for RGB, 16 -> 32 -> 64 -> 128 channels)
        Use adaptive average pooling to reduce the spatial dimensions before classification. (1x1 output for each channel, total 128 channels)
        Finally, flatten the features and apply dropout for regularization before the final linear layer for classification.
        '''
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)


class CNNLSTMOrdinalClassifier(nn.Module):
    """CNN feature extractor followed by an LSTM over spectrogram windows."""

    def __init__(self, num_classes=3, hidden_size=128, num_layers=1,
                 dropout=0.35):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Ordinal classification requires at least two classes")
        self.features = SmallEEGCNN(num_classes=128, dropout=dropout).features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.temporal = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes - 1),
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("CNN-LSTM input must have shape [batch, time, channels, height, width]")
        batch_size, sequence_length = x.shape[:2]
        frame_features = self.features(x.reshape(batch_size * sequence_length, *x.shape[2:]))
        frame_features = self.global_pool(frame_features).flatten(1)
        frame_features = frame_features.reshape(batch_size, sequence_length, -1)
        sequence_features, _ = self.temporal(frame_features)
        return self.classifier(sequence_features[:, sequence_length // 2])


def build_cnn_lstm_ordinal_classifier(num_classes=3, hidden_size=128,
                                      num_layers=1, dropout=0.35):
    """Build an ordinal CNN-LSTM for fixed-length spectrogram sequences."""
    return CNNLSTMOrdinalClassifier(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


def build_small_cnn_classifier(num_classes=2, dropout=0.35):
    """Build the compact CNN used by the Fz-only execution pipeline."""
    return SmallEEGCNN(num_classes=num_classes, dropout=dropout)


def build_small_cnn_ordinal_classifier(num_classes=3, dropout=0.35):
    """Build a cumulative ordinal classifier.

    For K ordered classes, the head has K-1 outputs. For the three PERCLOS
    classes these outputs represent PERCLOS >= 0.35 and PERCLOS > 0.70.
    """
    if num_classes < 2:
        raise ValueError("Ordinal classification requires at least two classes")
    return SmallEEGCNN(num_classes=num_classes - 1, dropout=dropout)


def ordinal_targets(labels, num_classes=3):
    """Convert class indices [0, ..., K-1] to cumulative binary targets."""
    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds).float()


def ordinal_logits_to_classes(logits, thresholds=(0.5, 0.5)):
    """Convert cumulative ordinal probabilities to class indices using 2 thresholds.
    
    thresholds[0]: threshold for P(class >= 1) [Transition or Drowsy]
    thresholds[1]: threshold for P(class >= 2) [Drowsy]
    """
    probabilities = torch.sigmoid(logits)
    t = torch.as_tensor(thresholds, device=logits.device, dtype=probabilities.dtype)
    return (probabilities >= t).sum(dim=1).long()


def train_ordinal_model(model, train_loader, val_loader, num_classes=3,
                        epochs=25, lr=3e-4, device="cuda",
                        early_stopping_patience=5):
    """Train a cumulative ordinal model with threshold-balanced BCE loss."""
    model = model.to(device)
    train_labels = _dataset_labels(train_loader.dataset)
    if set(train_labels.tolist()) != set(range(num_classes)):
        raise ValueError("Training split must contain every ordinal class")

    # Each cumulative threshold has a different imbalance. Threshold 2 is
    # positive only for Drowsy, so use a separate positive weight for it.
    cumulative = np.stack([
        train_labels >= threshold for threshold in range(1, num_classes)
    ], axis=1)
    positive = cumulative.sum(axis=0).astype(np.float64)
    negative = cumulative.shape[0] - positive
    pos_weight = torch.tensor(negative / np.maximum(positive, 1),
                              dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    feature_params = [p for p in model.features.parameters() if p.requires_grad]
    temporal_params = [
        p for p in getattr(model, "temporal", nn.Identity()).parameters()
        if p.requires_grad
    ]
    optimizer = torch.optim.Adam([
        {"params": feature_params, "lr": lr * 0.1},
        {"params": temporal_params, "lr": lr},
        {"params": model.classifier.parameters(), "lr": lr},
    ], weight_decay=1e-4)
    print(f"Ordinal threshold positive counts: {positive.astype(int).tolist()}; "
          f"pos weights: {pos_weight.detach().cpu().numpy().round(3).tolist()}")

    best_balanced_acc = -float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            targets = ordinal_targets(labels, num_classes=num_classes)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            predictions = ordinal_logits_to_classes(logits.detach())
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        model.eval()
        val_predictions, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(device))
                predictions = ordinal_logits_to_classes(logits)
                val_predictions.extend(predictions.cpu().tolist())
                val_labels.extend(labels.tolist())

        per_class_recall = [
            sum(p == y == class_id for p, y in zip(val_predictions, val_labels)) /
            max(1, sum(y == class_id for y in val_labels))
            for class_id in range(num_classes)
        ]
        val_balanced_acc = float(np.mean(per_class_recall))
        val_acc = float(np.mean(np.asarray(val_predictions) == np.asarray(val_labels)))
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / total:.4f} - "
              f"Train Acc: {correct / total:.4f} - Val Acc: {val_acc:.4f} - "
              f"Val Balanced Acc: {val_balanced_acc:.4f}")

        if val_balanced_acc > best_balanced_acc:
            best_balanced_acc = val_balanced_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}; restoring epoch "
                      f"{best_epoch} (best validation balanced accuracy: "
                      f"{best_balanced_acc:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def _dataset_labels(dataset):
    """Read labels without loading/transformation of every spectrogram."""
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        return np.asarray(dataset.dataset.labels)[np.asarray(dataset.indices)]
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))])


def train_vgg16(model, train_loader, val_loader, epochs=8, lr=1e-4, device='cuda',
                early_stopping_patience=2, class_weight_power=1.0):
    model = model.to(device)
    train_labels = _dataset_labels(train_loader.dataset)
    num_classes = int(model.classifier[-1].out_features)
    counts = np.bincount(train_labels, minlength=num_classes)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training split is missing class(es): {missing}")
    # Counteract class imbalance so the model does not simply predict the
    # majority PERCLOS class.
    inverse_frequency = len(train_labels) / (num_classes * counts)
    class_weights = inverse_frequency ** class_weight_power
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    feature_params = [p for p in model.features.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([
        {"params": feature_params, "lr": lr * 0.1},
        {"params": model.classifier.parameters(), "lr": lr},
    ], weight_decay=1e-4)
    print(f"Training class counts: {counts.tolist()}; "
          f"loss weights: {class_weights.round(3).tolist()}")
    best_balanced_acc = -float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_predictions, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_predictions.extend(predicted.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_acc = val_correct / val_total
        per_class_recall = [
            sum(p == y == class_id for p, y in zip(val_predictions, val_labels)) /
            max(1, sum(y == class_id for y in val_labels))
            for class_id in range(num_classes)
        ]
        balanced_acc = float(np.mean(per_class_recall))
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - "
              f"Train Acc: {epoch_acc:.4f} - Val Acc: {val_acc:.4f} - "
              f"Val Balanced Acc: {balanced_acc:.4f}")

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}; restoring epoch "
                      f"{best_epoch} (best validation balanced accuracy: "
                      f"{best_balanced_acc:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
