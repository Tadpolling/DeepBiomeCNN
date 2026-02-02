import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class DeepBiomeCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepBiomeCNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer5 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.global_pool(x)
        embedding = x.view(x.size(0), -1)
        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.fc(normalized_embedding)
        return normalized_embedding, logits

class VerticalCrop:
    def __call__(self, img):
        top = int(img.height * 0.1)
        bottom = int(img.height * 0.9)
        return TF.crop(img, top=top, left=0, height=bottom - top, width=img.width)

def evaluate_per_class(model, loader, classes, device):
    model.eval()
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    print("\n--- Per-Class Performance ---")
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, logits = model(images)
            _, predictions = torch.max(logits, 1)
            
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # Print Report
    print(f"{'Biome Class':<20} | {'Accuracy':<10} | {'Count':<10}")
    print("-" * 45)
    for classname, correct_count in correct_pred.items():
        total_count = total_pred[classname]
        if total_count > 0:
            accuracy = 100 * float(correct_count) / total_count
            print(f"{classname:<20} | {accuracy:>8.2f}% | {total_count:>5}")
        else:
            print(f"{classname:<20} |   N/A      |     0")

def plot_confusion_matrix(model, loader, classes, device):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, logits = model(images)
            _, preds = torch.max(logits, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.title("Biome Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig('evaluation_confusion_matrix.png')
    plt.show()
    print("Confusion matrix saved to evaluation_confusion_matrix.png")

if __name__ == "__main__":
    # Settings
    DATASET_PATH = '../data/Train Sets'  # Check this path matches yours
    MODEL_PATH = "DeepBiomeCNN.pth"
    BATCH_SIZE = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([
        VerticalCrop(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # Using normalization data
        transforms.Normalize([0.3307, 0.3753, 0.4310], [0.2531, 0.2555, 0.2851]) 
    ])

    full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=data_transforms)
    class_names = full_dataset.classes
    
    # Replicate the split
    generator = torch.Generator().manual_seed(42)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # We only need the test indices here
    indices = torch.randperm(total_size, generator=generator).tolist()
    test_indices = indices[train_size + val_size:]
    
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loaded {len(test_dataset)} test images.")

    model = DeepBiomeCNN(num_classes=len(class_names)).to(device)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded weights from {MODEL_PATH}")
        print(f"Model reached {checkpoint.get('best_acc', 0):.2f}% validation accuracy during training.")
    else:
        print(f"Error: Could not find {MODEL_PATH}")
        exit()

    evaluate_per_class(model, test_loader, class_names, device)
    plot_confusion_matrix(model, test_loader, class_names, device)