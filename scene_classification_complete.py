# %% [markdown]
# # Scene Classification using ResNet50, EfficientNet-B0, and ViT
# **Problem Statement:** Classify scenes (beach, street, office, forest) using ResNet, EfficientNet, ViT with Places365/SUN dataset.

# %% --- Cell 1: Install Dependencies ---
!pip install torch torchvision timm matplotlib scikit-learn tqdm

# %% --- Cell 2: Mount Google Drive & Setup ---
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Navigate to your project folder
project_path = "/content/drive/My Drive/VRSU Project"
os.chdir(project_path)
print(f"Working directory: {os.getcwd()}")

# %% --- Cell 3: Verify Dataset ---
import os

# Check dataset structure
if os.path.exists("dataset/train/train"):
    print("✅ Found dataset/train/train")
    print("Classes:", os.listdir("dataset/train/train")[:10])
else:
    print("❌ dataset/train/train NOT FOUND!")
    print("Current directory contents:", os.listdir())

# %% --- Cell 4: Create 4-Class Subset ---
import os
import shutil

source_train = "dataset/train/train"
source_test = "dataset/test/test"

target_base = "dataset_4class"
target_train = os.path.join(target_base, "train")
target_test = os.path.join(target_base, "test")

classes_needed = ["beach", "forest_broadleaf", "office", "street"]

# Create folders
for split in ["train", "test"]:
    for cls in classes_needed:
        os.makedirs(os.path.join(target_base, split, cls), exist_ok=True)

# Copy training data
for cls in classes_needed:
    shutil.copytree(
        os.path.join(source_train, cls),
        os.path.join(target_train, cls),
        dirs_exist_ok=True
    )

# Copy testing data
for cls in classes_needed:
    shutil.copytree(
        os.path.join(source_test, cls),
        os.path.join(target_test, cls),
        dirs_exist_ok=True
    )

print("✅ 4-class dataset created successfully!")

# Rename forest_broadleaf to forest
if os.path.exists("dataset_4class/train/forest_broadleaf"):
    os.rename("dataset_4class/train/forest_broadleaf", "dataset_4class/train/forest")
    os.rename("dataset_4class/test/forest_broadleaf", "dataset_4class/test/forest")
    print("✅ Renamed forest_broadleaf → forest")

print("Train classes:", sorted(os.listdir("dataset_4class/train")))
print("Test classes:", sorted(os.listdir("dataset_4class/test")))

# %% --- Cell 5: Data Loading with Transforms ---
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ImageNet normalization (important for pretrained models)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

train_dataset = datasets.ImageFolder("dataset_4class/train", transform=train_transform)
test_dataset = datasets.ImageFolder("dataset_4class/test", transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class_names = train_dataset.classes
num_classes = len(class_names)

print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% --- Cell 6: Training & Evaluation Functions ---
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5, model_name="Model"):
    """Train a model and return training history."""
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluate after each epoch
        val_acc = evaluate_model(model, test_loader, device)
        history["train_loss"].append(running_loss)
        history["val_acc"].append(val_acc)

        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs} — Loss: {running_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return history


def evaluate_model(model, loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def get_predictions(model, loader, device):
    """Get all predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(model, loader, device, class_names, model_name):
    """Plot confusion matrix for a model."""
    preds, labels = get_predictions(model, loader, device)
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Print classification report
    print(f"\n📊 Classification Report — {model_name}")
    print(classification_report(labels, preds, target_names=class_names))

    return preds, labels

# %% [markdown]
# ---
# ## Model 1: ResNet50

# %% --- Cell 7: ResNet50 — Build & Train ---
from torchvision.models import resnet50, ResNet50_Weights

# Build model
resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model = resnet_model.to(device)

# Optimizer & Loss
resnet_criterion = nn.CrossEntropyLoss()
resnet_optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.0001)

# Train
print("=" * 60)
print("🔵 Training ResNet50")
print("=" * 60)
resnet_history = train_model(
    resnet_model, train_loader, resnet_criterion, resnet_optimizer,
    device, num_epochs=5, model_name="ResNet50"
)

# %% --- Cell 8: ResNet50 — Evaluate & Confusion Matrix ---
print("=" * 60)
print("🔵 Evaluating ResNet50")
print("=" * 60)
resnet_acc = evaluate_model(resnet_model, test_loader, device)
print(f"\n✅ ResNet50 Final Test Accuracy: {resnet_acc:.2f}%\n")
plot_confusion_matrix(resnet_model, test_loader, device, class_names, "ResNet50")

# Save model
torch.save(resnet_model.state_dict(), "resnet50_scene_model.pth")
print("💾 Saved: resnet50_scene_model.pth")

# %% [markdown]
# ---
# ## Model 2: EfficientNet-B0

# %% --- Cell 9: EfficientNet-B0 — Build & Train ---
import timm

# Build model
effnet_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
effnet_model = effnet_model.to(device)

# Optimizer & Loss
effnet_criterion = nn.CrossEntropyLoss()
effnet_optimizer = torch.optim.Adam(effnet_model.parameters(), lr=0.0001)

# Train
print("=" * 60)
print("🟢 Training EfficientNet-B0")
print("=" * 60)
effnet_history = train_model(
    effnet_model, train_loader, effnet_criterion, effnet_optimizer,
    device, num_epochs=5, model_name="EfficientNet-B0"
)

# %% --- Cell 10: EfficientNet-B0 — Evaluate & Confusion Matrix ---
print("=" * 60)
print("🟢 Evaluating EfficientNet-B0")
print("=" * 60)
effnet_acc = evaluate_model(effnet_model, test_loader, device)
print(f"\n✅ EfficientNet-B0 Final Test Accuracy: {effnet_acc:.2f}%\n")
plot_confusion_matrix(effnet_model, test_loader, device, class_names, "EfficientNet-B0")

# Save model
torch.save(effnet_model.state_dict(), "efficientnet_b0_scene_model.pth")
print("💾 Saved: efficientnet_b0_scene_model.pth")

# %% [markdown]
# ---
# ## Model 3: ViT (Vision Transformer)

# %% --- Cell 11: ViT-Base — Build & Train ---
import timm

# Build model
vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
vit_model = vit_model.to(device)

# Optimizer & Loss
vit_criterion = nn.CrossEntropyLoss()
vit_optimizer = torch.optim.Adam(vit_model.parameters(), lr=0.0001)

# Train
print("=" * 60)
print("🟣 Training ViT-Base-Patch16-224")
print("=" * 60)
vit_history = train_model(
    vit_model, train_loader, vit_criterion, vit_optimizer,
    device, num_epochs=5, model_name="ViT-Base"
)

# %% --- Cell 12: ViT-Base — Evaluate & Confusion Matrix ---
print("=" * 60)
print("🟣 Evaluating ViT-Base-Patch16-224")
print("=" * 60)
vit_acc = evaluate_model(vit_model, test_loader, device)
print(f"\n✅ ViT-Base Final Test Accuracy: {vit_acc:.2f}%\n")
plot_confusion_matrix(vit_model, test_loader, device, class_names, "ViT-Base")

# Save model
torch.save(vit_model.state_dict(), "vit_base_scene_model.pth")
print("💾 Saved: vit_base_scene_model.pth")

# %% [markdown]
# ---
# ## Model Comparison

# %% --- Cell 13: Comparison Table ---
print("=" * 60)
print("📊 MODEL COMPARISON — Scene Classification (4 Classes)")
print("=" * 60)
print(f"{'Model':<25} {'Test Accuracy':>15}")
print("-" * 42)
print(f"{'ResNet50':<25} {resnet_acc:>14.2f}%")
print(f"{'EfficientNet-B0':<25} {effnet_acc:>14.2f}%")
print(f"{'ViT-Base-Patch16-224':<25} {vit_acc:>14.2f}%")
print("-" * 42)

# Find best model
best_name = ["ResNet50", "EfficientNet-B0", "ViT-Base"][
    [resnet_acc, effnet_acc, vit_acc].index(max(resnet_acc, effnet_acc, vit_acc))
]
print(f"\n🏆 Best Model: {best_name} ({max(resnet_acc, effnet_acc, vit_acc):.2f}%)")

# %% --- Cell 14: Accuracy Bar Chart ---
import matplotlib.pyplot as plt

model_names = ["ResNet50", "EfficientNet-B0", "ViT-Base"]
accuracies = [resnet_acc, effnet_acc, vit_acc]
colors = ['#4285F4', '#34A853', '#9C27B0']

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=0.8)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.title("Scene Classification — Model Accuracy Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy (%)", fontsize=12)
plt.ylim(0, 105)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %% --- Cell 15: Training Loss Comparison ---
import matplotlib.pyplot as plt

epochs_range = range(1, 6)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, resnet_history["train_loss"], 'o-', label="ResNet50", color='#4285F4')
plt.plot(epochs_range, effnet_history["train_loss"], 's-', label="EfficientNet-B0", color='#34A853')
plt.plot(epochs_range, vit_history["train_loss"], '^-', label="ViT-Base", color='#9C27B0')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, resnet_history["val_acc"], 'o-', label="ResNet50", color='#4285F4')
plt.plot(epochs_range, effnet_history["val_acc"], 's-', label="EfficientNet-B0", color='#34A853')
plt.plot(epochs_range, vit_history["val_acc"], '^-', label="ViT-Base", color='#9C27B0')
plt.title("Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(alpha=0.3)

plt.suptitle("Scene Classification — Training Curves Comparison", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Summary
# - **ResNet50**: Classic CNN architecture, fast training, strong baseline
# - **EfficientNet-B0**: Efficient scaling, good accuracy with fewer parameters
# - **ViT-Base**: Transformer-based, captures global context, may need more data to shine
#
# All models were fine-tuned from ImageNet pretrained weights on a 4-class subset
# (beach, forest, office, street) from the Places365/SUN dataset.
