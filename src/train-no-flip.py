import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
import torchvision.transforms.functional as TF


import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

import os

import random

# Creates masks that remove a part of the model, either from the left, right or center
class MinecraftSpecificMask:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        # img is a Tensor [C, H, W]
        _, h, w = img.shape
        
        # Define mask types: 0=None, 1=Right Hand, 2=Left Hand, 3=Center (Player)
        mask_choice = random.choice([1, 2, 3])

        if mask_choice == 1:
            # Right Hand: Bottom Right
            # Minimal mask: roughly the last 20% of width and bottom 30% of height
            img[:, int(h*0.7):, int(w*0.8):] = 0
            
        elif mask_choice == 2:
            # Left Hand (Off-hand): Bottom Left
            img[:, int(h*0.7):, :int(w*0.2)] = 0
            
        elif mask_choice == 3:
            # Center (Player in F5/F3 view)
            # Minimal mask: Center 20% of the screen
            start_h, end_h = int(h*0.4), int(h*0.6)
            start_w, end_w = int(w*0.4), int(w*0.6)
            img[:, start_h:end_h, start_w:end_w] = 0

        return img

# Balances the images so that different biomes are always represented.
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes, n_samples):
        # labels: list or tensor of all labels in the dataset
        # n_classes: number of unique classes per batch (P)
        # n_samples: number of images per class (K)
        self.labels = np.array(labels)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
            
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_classes * self.n_samples

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:
                               self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size

#Crops the top and bottom of the image slightly to remove UI 
class VerticalCrop:
    def __call__(self, img):
        # We crop the top 10% (UI) and the bottom 25% (where hands/items are most prominent)
        top = int(img.height * 0.1)
        bottom = int(img.height * 0.9)
        return TF.crop(
            img,
            top=top,
            left=0,
            height=bottom - top,
            width=img.width
        )

desired_size = (128, 128)

data_transforms = transforms.Compose([
    VerticalCrop(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

DATASET_PATH = '../data/Train Sets'

full_dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=data_transforms
)

print(full_dataset.classes)
print(len(full_dataset))

#Computes statistics on data
def compute_mean_std(dataloader):
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    total_pixels = 0

    for images, _ in dataloader:
        # images shape: [Batch, Channels, Height, Width]
        batch_samples = images.size(0)
        channels = images.size(1)
        h, w = images.size(2), images.size(3)

        # Count total pixels for this batch
        total_pixels += batch_samples * h * w

        # 1. Sum of all pixels (per channel)
        sum_ += images.sum(dim=[0, 2, 3])

        # 2. Sum of all squared pixels (per channel)
        sum_sq += (images**2).sum(dim=[0, 2, 3])

    # Final Mean
    mean = sum_ / total_pixels

    # Final Std: sqrt( (Sum of Squares / Total Pixels) - (Mean^2) )
    std = torch.sqrt((sum_sq / total_pixels) - (mean**2))

    return mean, std

# Creating train-test split
generator=torch.Generator().manual_seed(42)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

indices = torch.randperm(total_size, generator=generator).tolist()

train_indices = indices[:train_size]
val_indices = indices[train_size : train_size + val_size]
test_indices = indices[train_size + val_size:]

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

print(f"Total: {total_size} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")


cores = os.cpu_count()

BATCH_SIZE = 256
NUM_WORKERS = 2
train_labels = [full_dataset.targets[i] for i in train_indices]
train_sampler = BalancedBatchSampler(train_labels, n_classes=len(full_dataset.classes), n_samples=3)
train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True
    )

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print(f"Training with Batch Size: {BATCH_SIZE} and Workers: {NUM_WORKERS}")

mean, std = compute_mean_std(train_loader)
print(mean)
print(std)

normalize = transforms.Normalize(mean=mean, std=std)

# Training augments includes all the data
train_transform = transforms.Compose([
    VerticalCrop(),
    transforms.ToTensor(),
    MinecraftSpecificMask(p=0.6),
    normalize,
])

test_transform = transforms.Compose([
    VerticalCrop(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    normalize,
])

# Adding normalization 
train_loader.dataset.dataset.transform = train_transform
test_loader.dataset.dataset.transform  = test_transform

# Verifying validity of data
images, labels = next(iter(train_loader))
print(images.shape)  # [B, 3, 128, 128]
print(labels.shape)  # [B]


# Creating the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# # Optimization
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

class DeepBiomeCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepBiomeCNN, self).__init__()

        # Layer 1-5 (Keeping your existing architecture)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(512, num_classes, bias = False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = self.global_pool(x)
        embedding = x.view(x.size(0), -1)
        
        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        
        # Pass through classification head
        logits = self.fc(normalized_embedding)
        
        return normalized_embedding, logits
    
class BiomeTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(BiomeTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # calculates the pairwise distance matrix
        # Since embeddings are normalized, dist = 2 - 2 * cos_sim
        dot_product = torch.matmul(embeddings, embeddings.t())
        
        # square_norm is 1.0 for normalized vectors, but we calculate 
        # it dynamically to be safe and flexible.
        square_norm = dot_product.diag()
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        
        # Guard against negative values from float precision errors
        distances = F.relu(distances) 

        # Identifies images of the same biome (excluding the image itself)
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_pos.fill_diagonal_(False)
        
        # Identifies images of different biomes
        mask_neg = labels.unsqueeze(0) != labels.unsqueeze(1)

        # hardest Positive Mining
        # Find the max distance between images of the same biome
        hardest_pos, _ = torch.max(distances * mask_pos.float(), dim=1)

        # hardest Negative Mining
        # Find the min distance between images of different biomes
        # We add a large value to positive pairs so they aren't picked as 'min'
        max_dist = distances.max(dim=1, keepdim=True)[0]
        distances_for_neg = distances + max_dist * (~mask_neg).float()
        hardest_neg, _ = torch.min(distances_for_neg, dim=1)

        # final Loss
        loss = F.relu(hardest_pos - hardest_neg + self.margin)

        return loss.mean()


MODEL_PATH = "DeepBiomeCNN.pth"
num_classes = len(full_dataset.classes)
model = DeepBiomeCNN(num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion_cls = nn.CrossEntropyLoss()
criterion_tri = BiomeTripletLoss(margin=0.3)

# Initialize history lists
history = {
    'train_loss': [], 'train_acc': [], 
    'val_loss': [], 'val_acc': [], 
    'interclass_dist': []
}
best_val_acc = 0.0
patience_counter = 0
start_epoch = 0

if os.path.exists(MODEL_PATH):
    print(f"--- Loading existing model from {MODEL_PATH} ---")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    history = checkpoint['history']
    best_val_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    print(f"Resuming from Epoch {start_epoch} (Best Val Acc: {best_val_acc:.2f}%)")


# Training

PATIENCE = 10
triplet_weight= 0.1 
Warmup_length =10
print(f"--- Starting Training (Patience: {PATIENCE}) ---")
max_epochs = 200 # Set high, early stopping will handle exit
for epoch in range(start_epoch, max_epochs):
    model.train()
    t_loss, t_correct, t_total = 0, 0, 0
    
    if epoch == Warmup_length + PATIENCE//2:
        best_val_acc =0.0
        print("reset best validation accuracy after warmup ended")

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        emb, logits = model(images)
        alpha = triplet_weight if epoch > Warmup_length else 0.0
        loss = criterion_cls(logits, labels) + (alpha * criterion_tri(emb, labels))
        loss.backward()
        optimizer.step()
        
        t_loss += loss.item()
        t_correct += logits.max(1)[1].eq(labels).sum().item()
        t_total += labels.size(0)

    # Validation
    model.eval()
    v_loss, v_correct, v_total, all_embs, all_lbls = 0, 0, 0, [], []
    with torch.no_grad():
        for img, lbl in val_loader:
            img, lbl = img.to(device), lbl.to(device)
            emb, logit = model(img)
            
            v_loss += (criterion_cls(logit, lbl) + (alpha * criterion_tri(emb, lbl))).item()
            v_correct += logit.max(1)[1].eq(lbl).sum().item()
            v_total += lbl.size(0)
            all_embs.append(emb); all_lbls.append(lbl)

    # Compute Metrics
    cur_val_acc = 100 * v_correct / v_total
    cur_train_acc = 100 * t_correct / t_total
    
    # Separation Distance
    concat_emb = torch.cat(all_embs); concat_lbl = torch.cat(all_lbls)
    centroids = torch.stack([concat_emb[concat_lbl == l].mean(dim=0) for l in torch.unique(concat_lbl)])
    cur_dist = torch.cdist(centroids, centroids).mean().item()

    history['train_loss'].append(t_loss / len(train_loader))
    history['train_acc'].append(cur_train_acc)
    history['val_loss'].append(v_loss / len(val_loader))
    history['val_acc'].append(cur_val_acc)
    history['interclass_dist'].append(cur_dist)

    print(f"Epoch {epoch+1}: Train Acc: {cur_train_acc:.1f}% | Val Acc: {cur_val_acc:.1f}% | Dist: {cur_dist:.3f}")

    # EARLY STOPPING LOGIC
    if cur_val_acc > best_val_acc:
        best_val_acc = cur_val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_val_acc,
            'history': history
        }, MODEL_PATH)
        print("Improvement found! Best model saved.")
    else:
        patience_counter += 1
        print(f" No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered. Loading best weights")
            # Restore best weights for final graphs/test
            checkpoint = torch.load(MODEL_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            break

def evaluate_on_test_set(model, test_loader):
    """Calculates final classification accuracy on the held-out test set."""
    model.eval()
    t_correct = 0
    t_total = 0
    
    print("\n--- Running Final Evaluation on Test Set ---")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            _, logits = model(images)
            
            _, predicted = torch.max(logits.data, 1)
            t_total += labels.size(0)
            t_correct += (predicted == labels).sum().item()
            
    test_acc = 100 * t_correct / t_total
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    return test_acc

def save_and_plot_report(history, model, loader, classes):
    # Set global plotting style for a "Scientific" look
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16
    })

    epochs_range = range(1, len(history['val_acc']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300) # Higher DPI for print quality

    # Loss Plot (Combined CE + Triplet)
    axes[0].plot(epochs_range, history['train_loss'], label='Training', color='#1f77b4', linewidth=1.5)
    axes[0].plot(epochs_range, history['val_loss'], label='Validation', color='#ff7f0e', linestyle='--', linewidth=1.5)
    axes[0].set_title('Loss Convergence')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Total Objective Loss')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(frameon=True, loc='upper right')

    # Accuracy Plot
    axes[1].plot(epochs_range, history['train_acc'], label='Training', color='#2ca02c', linewidth=1.5)
    axes[1].plot(epochs_range, history['val_acc'], label='Validation', color='#d62728', linestyle='--', linewidth=1.5)
    axes[1].set_title('Classification Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(frameon=True, loc='lower right')

    # 3. Embedding Separation (The "Science" of Triplet Loss)
    axes[2].plot(epochs_range, history['interclass_dist'], color='#9467bd', linewidth=2)
    axes[2].set_title('Mean Inter-class Centroid Distance')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Euclidean Distance ($L_2$)')
    axes[2].grid(True, linestyle=':', alpha=0.6)

    # Global layout adjustments
    plt.tight_layout(pad=3.0)
    
    # Save as PDF or PNG (PDF is preferred for LaTeX/scientific papers)
    plt.savefig('biome_training_report.pdf', bbox_inches='tight')
    plt.savefig('biome_training_report.png', bbox_inches='tight')
    plt.show()

final_accuracy = evaluate_on_test_set(model, test_loader)

save_and_plot_report(history, model, val_loader, full_dataset.classes)
