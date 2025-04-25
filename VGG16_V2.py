"""
Modifications ResNet :
    Add One-hot encoding into DeepFashionMultilabelDataset class
    Change CrossEntropyLoss() to BCEWithLogitsLoss() because our dataset is multiple-label
    Add parameter freezing and more trainable parameters in MultiOutputResNet class
    Increase data volume by using RandomHorizontalFlip() and ColorJitter()
    Add label distribution check : alert highly imbalanced dataset
    Add pos_weight to resolve dataset imbalance problem
    Add tiny subsample overfitting test : Pass !
    Problem remained : Precision metrics too low, model infers lots of false-positive

Version 1 Modifications VGG :
    Change base model to VGG16 instead of ResNet
    Unfreezing 26 and 28 layer of VGG16 model to release more trainable parameters
    Update model based on Accuracy instead of val_loss
    Threshold searching for multiple labels based on F1-score
    Problem remained : The imbalanced data problem still hasn't been solved, recall metrics stay low

Version 2 Modifications VGG :
    Add Focal Loss instead of pos_weight
    Add Hybrid Sampling using WeightedRandomSampler
    Update model based on F1-score
    Evaluate model by F1-score
    Early stop depends on F1-score with the best early_stop_patience=20

Version 3 Modifications VGG :
    lr searching, the best hyperparameters found : lr=1e-5
    Scheduler strategy searching, the best hyperparameters found : Using Step Decay scheduler with step_size=5 and gamma=0.2
    Regularization searching, the best hyperparameters found : weight_decay=1e-5 within Adam
    Add Main function to avoid training loop implementation during inference

Inference H&M (Annotation) : inf_hm.py

Cloth Recommendation : UUCF_ML.py

"""


import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.utils.data import random_split
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import glob
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from torch.utils.data import WeightedRandomSampler


# ============================== CONFIGURATION ============================== #
batch_size = 16
learning_rate = 1e-5
weight_decay = 1e-5
gamma = 0.2  # scheduler
step_size = 5  # scheduler step decay
gamma_cos = 0.8  # scheduler cosinus decay
num_epochs = 200
n_workers = 8  # Data loading speed control
model_name = 'VGG16_V2'


# ============================== TENSORBOARD AND CHECKPOINT SETUP============================== #
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = f"runs_pfe/{timestamp}_{model_name}"
writer = SummaryWriter(log_dir=log_dir)

checkpoint_dir = f"checkpoints_{model_name}"
os.makedirs(checkpoint_dir, exist_ok=True)


# ============================== TRAINING PREPARATION============================== #
# Initialize tracking best model
best_val_f1 = 0
early_stop_patience = 20
early_stop_counter = 0

# Device define
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used is : {device}")

# Load base model
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
num_ftrs = vgg.classifier[0].in_features  # VGG16 uses classifier, not fc


# ============================== CLASS AND FUNCTION ============================== #
class DeepFashionMultilabelDataset(Dataset):
    def __init__(self, img_dir, fabric_file, shape_file, pattern_file, transform=None):
        self.img_dir = img_dir
        self.image_paths = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

        # Read raw labels
        raw_fabric = self._read_fabric_label_file(fabric_file)
        raw_shape = self._read_shape_label_file(shape_file)
        raw_pattern = self._read_pattern_label_file(pattern_file)

        # Fix shape label mismatch
        while len(raw_shape) < len(raw_fabric):
            raw_shape.append([0] * 12)

        # Sanity check
        assert len(self.image_paths) == len(raw_fabric) == len(raw_shape) == len(raw_pattern), \
            "Mismatch between images and label counts."

        # Save raw labels (optional for debugging)
        self.raw_fabric = raw_fabric
        self.raw_shape = raw_shape
        self.raw_pattern = raw_pattern

        # One-hot encode once and store as tensors
        self.fabric_labels = [self._one_hot_encode(lbl, 8) for lbl in raw_fabric]  # 3 x 8 = 24
        self.pattern_labels = [self._one_hot_encode(lbl, 8) for lbl in raw_pattern]  # 3 x 8 = 24
        self.shape_labels = [self._one_hot_encode_shape(lbl) for lbl in raw_shape]  # 50-dim

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        fabric_label = self.fabric_labels[idx]
        pattern_label = self.pattern_labels[idx]
        shape_label = self.shape_labels[idx]

        return image, shape_label, fabric_label, pattern_label

    def _read_fabric_label_file(self, fabric_file):
        labels = []
        with open(fabric_file, 'r') as f:
            for line in f:
                parts = list(map(int, line.strip().split()[1:4]))
                labels.append(parts)
        return labels

    def _read_pattern_label_file(self, pattern_file):
        labels = []
        with open(pattern_file, 'r') as f:
            for line in f:
                parts = list(map(int, line.strip().split()[1:4]))
                labels.append(parts)
        return labels

    def _read_shape_label_file(self, shape_file):
        labels = []
        with open(shape_file, 'r') as f:
            for line in f:
                parts = list(map(int, line.strip().split()[1:13]))
                labels.append(parts)
        return labels

    def _one_hot_encode(self, labels, num_classes):
        """One-hot encode 3-label list into a flat 24-dim tensor."""
        return torch.cat([
            F.one_hot(torch.tensor(label), num_classes).float() for label in labels
        ])

    def _one_hot_encode_shape(self, shape_label):
        """One-hot encode 12 shape attributes into a flat 50-dim tensor."""
        shape_categories = [6, 5, 4, 3, 5, 3, 3, 3, 5, 7, 3, 3]
        return torch.cat([
            F.one_hot(torch.tensor(shape_label[i]), num_classes=shape_categories[i]).float()
            for i in range(12)
        ])


class MultiOutputVGG(nn.Module):
    def __init__(self, base_model, num_ftrs):
        super(MultiOutputVGG, self).__init__()
        # Use feature extractor part of VGG16
        self.base_model = base_model.features

        for name, param in vgg.features.named_parameters():
            if any(layer in name for layer in ['26', '28']):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.avgpool = base_model.avgpool  # VGG16 has a separate avgpool layer
        self.flatten = nn.Flatten()

        self.shape_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 50)
        )
        self.fabric_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 24)
        )
        self.pattern_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 24)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.shape_head(x), self.fabric_head(x), self.pattern_head(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return loss.mean() if self.reduction == 'mean' else loss


def calculate_loss_and_accuracy(loader, model, shape_criterion, fabric_criterion, pattern_criterion, device, thresholds=None, tune=False):
    model.eval()
    running_loss = 0.0
    correct_shape, correct_fabric, correct_pattern = 0, 0, 0
    total_samples = 0

    all_shape_true, all_shape_probs = [], []
    all_fabric_true, all_fabric_probs = [], []
    all_pattern_true, all_pattern_probs = [], []

    shape_thresh_eval = thresholds.get("shape") if thresholds else np.full((50,), 0.5)
    fabric_thresh_eval = thresholds.get("fabric") if thresholds else np.full((24,), 0.5)
    pattern_thresh_eval = thresholds.get("pattern") if thresholds else np.full((24,), 0.5)

    with torch.no_grad():
        for images, shape_labels, fabric_labels, pattern_labels in loader:
            images = images.to(device)
            shape_labels = shape_labels.float().to(device)
            fabric_labels = fabric_labels.float().to(device)
            pattern_labels = pattern_labels.float().to(device)

            shape_out, fabric_out, pattern_out = model(images)

            loss_shape = shape_criterion(shape_out, shape_labels)
            loss_fabric = fabric_criterion(fabric_out, fabric_labels)
            loss_pattern = pattern_criterion(pattern_out, pattern_labels)
            total_loss = loss_shape + loss_fabric + loss_pattern

            running_loss += total_loss.item()

            shape_prob = torch.sigmoid(shape_out)
            fabric_prob = torch.sigmoid(fabric_out)
            pattern_prob = torch.sigmoid(pattern_out)

            shape_pred = (shape_prob > torch.tensor(shape_thresh_eval).to(device)).float()
            fabric_pred = (fabric_prob > torch.tensor(fabric_thresh_eval).to(device)).float()
            pattern_pred = (pattern_prob > torch.tensor(pattern_thresh_eval).to(device)).float()

            correct_shape += (shape_pred == shape_labels).sum().item()
            correct_fabric += (fabric_pred == fabric_labels).sum().item()
            correct_pattern += (pattern_pred == pattern_labels).sum().item()

            all_shape_true.append(shape_labels.cpu())
            all_shape_probs.append(shape_prob.cpu())
            all_fabric_true.append(fabric_labels.cpu())
            all_fabric_probs.append(fabric_prob.cpu())
            all_pattern_true.append(pattern_labels.cpu())
            all_pattern_probs.append(pattern_prob.cpu())

            total_samples += images.size(0)

    # Convert to numpy
    shape_true = torch.cat(all_shape_true).numpy()
    shape_probs = torch.cat(all_shape_probs).numpy()
    fabric_true = torch.cat(all_fabric_true).numpy()
    fabric_probs = torch.cat(all_fabric_probs).numpy()
    pattern_true = torch.cat(all_pattern_true).numpy()
    pattern_probs = torch.cat(all_pattern_probs).numpy()

    # For reporting metrics — use fixed 0.5 threshold
    shape_pred = (shape_probs > 0.5).astype(int)
    fabric_pred = (fabric_probs > 0.5).astype(int)
    pattern_pred = (pattern_probs > 0.5).astype(int)

    shape_prec = precision_score(shape_true, shape_pred, average='macro', zero_division=0)
    shape_rec = recall_score(shape_true, shape_pred, average='macro', zero_division=0)
    fabric_prec = precision_score(fabric_true, fabric_pred, average='macro', zero_division=0)
    fabric_rec = recall_score(fabric_true, fabric_pred, average='macro', zero_division=0)
    pattern_prec = precision_score(pattern_true, pattern_pred, average='macro', zero_division=0)
    pattern_rec = recall_score(pattern_true, pattern_pred, average='macro', zero_division=0)

    avg_loss = running_loss / len(loader)
    shape_acc = correct_shape / (total_samples * 50)
    fabric_acc = correct_fabric / (total_samples * 24)
    pattern_acc = correct_pattern / (total_samples * 24)

    if tune:
        # Threshold tuning for saving the best model
        shape_tuned_thresh, shape_f1 = tune_threshold_and_f1(shape_true, shape_probs)
        fabric_tuned_thresh, fabric_f1 = tune_threshold_and_f1(fabric_true, fabric_probs)
        pattern_tuned_thresh, pattern_f1 = tune_threshold_and_f1(pattern_true, pattern_probs)
        avg_f1 = (shape_f1 + fabric_f1 + pattern_f1) / 3
    else:
        shape_tuned_thresh = shape_thresh_eval
        fabric_tuned_thresh = fabric_thresh_eval
        pattern_tuned_thresh = pattern_thresh_eval
        shape_f1 = f1_score(shape_true, shape_pred, average='macro', zero_division=0)
        fabric_f1 = f1_score(fabric_true, fabric_pred, average='macro', zero_division=0)
        pattern_f1 = f1_score(pattern_true, pattern_pred, average='macro', zero_division=0)
        avg_f1 = (shape_f1 + fabric_f1 + pattern_f1) / 3

    return (
        avg_loss, shape_acc, fabric_acc, pattern_acc,
        shape_prec, shape_rec, fabric_prec, fabric_rec, pattern_prec, pattern_rec,
        avg_f1, shape_f1, fabric_f1, pattern_f1,
        shape_tuned_thresh, fabric_tuned_thresh, pattern_tuned_thresh
    )


def save_checkpoint(epoch, model, optimizer, scheduler, val_accuracy, val_avg_f1, thresholds=None, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_accuracy': val_accuracy,
        'val_avg_f1': val_avg_f1,
        'learning_rate': scheduler.get_last_lr()[0],  # Save current LR
        'thresholds': thresholds
    }
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, path)

    if is_best:
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(checkpoint, best_model_path)
        print(
            f"✅ Saved best model at epoch {epoch + 1} (val_accuracy={val_accuracy:.4f}, learning rate={scheduler.get_last_lr()[0]})")


def tune_threshold_and_f1(y_true, y_probs):
    best_thresh = []
    best_f1s = []
    for i in range(y_true.shape[1]):
        f1_scores = []
        thresholds = np.linspace(0.1, 0.9, 9)
        for t in thresholds:
            y_probs = np.array(y_probs)
            pred = (y_probs[:, i] > t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            f1_scores.append(f1)
        best_idx = np.argmax(f1_scores)
        best_thresh.append(thresholds[best_idx])
        best_f1s.append(f1_scores[best_idx])
    avg_f1 = np.mean(best_f1s)
    return np.array(best_thresh), avg_f1


# ============================== MAIN FUNCTION ============================== #
if __name__ == "__main__":
    # ===============================Data Processing=================================#

    # Define image transforms for training by increasing data volume
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Paths to your dataset
    img_dir = '/array/shared/home/hwong/PFE/DeepFashion-Multimodal/images'
    fabric_file = '/array/shared/home/hwong/PFE/DeepFashion-Multimodal/labels/texture/fabric_ann.txt'
    shape_file = '/array/shared/home/hwong/PFE/DeepFashion-Multimodal/labels/shape/shape_anno_all.txt'
    pattern_file = '/array/shared/home/hwong/PFE/DeepFashion-Multimodal/labels/texture/pattern_ann.txt'

    # Instantiate the dataset and DataLoader
    dataset = DeepFashionMultilabelDataset(img_dir=img_dir,
                                           fabric_file=fabric_file,
                                           shape_file=shape_file,
                                           pattern_file=pattern_file,
                                           transform=transform)

    # ---------------- Label Distribution Check & pos_weight ---------------- #
    shape_tensor = torch.stack(dataset.shape_labels)  # [N, 50]
    fabric_tensor = torch.stack(dataset.fabric_labels)  # Shape: [N, 24]
    pattern_tensor = torch.stack(dataset.pattern_labels)  # Shape: [N, 24]

    #Compute Sample Weights (Hybrid Scoring)
    fabric_label_counts = fabric_tensor.sum(dim=0) + 1e-5
    pattern_label_counts = pattern_tensor.sum(dim=0) + 1e-5
    shape_label_counts = shape_tensor.sum(dim=0) + 1e-5

    fabric_weights = 1.0 / fabric_label_counts
    pattern_weights = 1.0 / pattern_label_counts
    shape_weights = 1.0 / shape_label_counts

    sample_weights = []
    for i in range(len(dataset)):
        fabric_score = (dataset.fabric_labels[i] * fabric_weights).sum()
        pattern_score = (dataset.pattern_labels[i] * pattern_weights).sum()
        shape_score = (dataset.shape_labels[i] * shape_weights).sum()
        total_score = (fabric_score + pattern_score + shape_score).item()
        sample_weights.append(total_score)


    shape_dist = shape_tensor.sum(dim=0)
    fabric_dist = fabric_tensor.sum(dim=0)
    pattern_dist = pattern_tensor.sum(dim=0)

    # Imbalance / Label Distribution Check
    """
    fabric_percent = fabric_dist / fabric_tensor.shape[0] * 100
    pattern_percent = pattern_dist / pattern_tensor.shape[0] * 100
    print("🧵 Fabric Label Distribution (%):", fabric_percent)
    print("🎨 Pattern Label Distribution (%):", pattern_percent)
    shape_categories = [6, 5, 4, 3, 5, 3, 3, 3, 5, 7, 3, 3]
    start = 0
    for i, cat_count in enumerate(shape_categories):
        group = shape_tensor[:, start:start + cat_count]  # Slice this attribute group
        group_dist = group.sum(dim=0)
        group_percent = group_dist / shape_tensor.shape[0] * 100
        print(f"🔸 Shape Attribute {i} Distribution (%): {group_percent}")
        start += cat_count
    """

    # Calculate pos_weight
    shape_pos_weight = (shape_tensor.shape[0] - shape_dist) / (shape_dist + 1e-5)
    fabric_pos_weight = (fabric_tensor.shape[0] - fabric_dist) / (fabric_dist + 1e-5)
    pattern_pos_weight = (pattern_tensor.shape[0] - pattern_dist) / (pattern_dist + 1e-5)

    # ---------------- Data split ---------------- #
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset into train, val, and test subsets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders for each split
    # Extract sample weights only for training set
    train_indices = train_dataset.indices if hasattr(train_dataset, 'indices') else range(len(train_dataset))
    train_sample_weights = [sample_weights[i] for i in train_indices]
    sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_sample_weights), replacement=True)
    #No need to shuffle, the sampler handles it
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=n_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)


    # ============================Training Setup============================#
    # Define the MultiOutputResNet model
    model = MultiOutputVGG(vgg, num_ftrs).to(device)

    # Define the criterion (loss function) for multi-output tasks
    # CrossEntropyLoss() is for single-label classification, but shape allows multiple categories per image.
    #shape_criterion = nn.BCEWithLogitsLoss(pos_weight=shape_pos_weight.to(device))
    #fabric_criterion = nn.BCEWithLogitsLoss(pos_weight=fabric_pos_weight.to(device))
    #pattern_criterion = nn.BCEWithLogitsLoss(pos_weight=pattern_pos_weight.to(device))

    shape_criterion = FocalLoss(alpha=1.0, gamma=2.0)
    fabric_criterion = FocalLoss(alpha=1.0, gamma=2.0)
    pattern_criterion = FocalLoss(alpha=1.0, gamma=2.0)

    # Optimizer and Scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                           weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # Step Decay
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_cos) #Continus Decay
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) #Continus Decay
    thresholds = {}

    # Resume From Checkpoint
    resume_training = True
    checkpoint_path = ""
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")))
    if checkpoint_files:
        checkpoint_path = checkpoint_files[-1]

    start_epoch = 0  # default if not resuming
    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['val_accuracy']
        best_val_f1  = checkpoint['val_avg_f1']
        lr = checkpoint.get('learning_rate', None)
        thresholds = checkpoint.get('thresholds', None)
        print(f"✅ Resumed from epoch {start_epoch}, best val F1: {best_val_f1 :.4f}, LR: {lr}")
    else:
        best_val_f1 = 0  # if not resuming

    # ---------------- Overfitting 10 Samples ---------------- #
    """
    tiny_dataset = torch.utils.data.Subset(train_dataset, range(10))
    tiny_loader = DataLoader(tiny_dataset, batch_size=2, shuffle=True)

    model.train()  # Set model to training mode

    for epoch in range(num_epochs):  # Train for a few epochs
        running_loss = 0.0
        correct_shape, correct_fabric, correct_pattern = 0, 0, 0
        total = 0

        for images, shape_labels, fabric_labels, pattern_labels in tiny_loader:
            images = images.to(device)
            shape_labels = shape_labels.float().to(device)
            fabric_labels = fabric_labels.float().to(device)
            pattern_labels = pattern_labels.float().to(device)

            optimizer.zero_grad()
            shape_output, fabric_output, pattern_output = model(images)

            loss_shape = shape_criterion(shape_output, shape_labels)
            loss_fabric = fabric_criterion(fabric_output, fabric_labels)
            loss_pattern = pattern_criterion(pattern_output, pattern_labels)
            total_loss = loss_shape + loss_fabric + loss_pattern

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            predicted_shape = (torch.sigmoid(shape_output) > 0.5).float()
            predicted_fabric = (torch.sigmoid(fabric_output) > 0.5).float()
            predicted_pattern = (torch.sigmoid(pattern_output) > 0.5).float()

            correct_shape += (predicted_shape == shape_labels).sum().item()
            correct_fabric += (predicted_fabric == fabric_labels).sum().item()
            correct_pattern += (predicted_pattern == pattern_labels).sum().item()

            total += images.size(0)

        avg_loss = running_loss / len(tiny_loader)
        shape_acc = correct_shape / (total * 50)
        fabric_acc = correct_fabric / (total * 24)
        pattern_acc = correct_pattern / (total * 24)
        avg_acc = (shape_acc + fabric_acc + pattern_acc) / 3

        print(f"[Overfit Epoch {epoch+1}] Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")
    """

    # ============================== TRAINING LOOP ============================== #
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(
            f"\n🔄 Running Epoch {epoch + 1}/{start_epoch + num_epochs} with learning rate {scheduler.get_last_lr()[0]}")

        model.train()
        running_loss = 0.0
        correct_shape, correct_fabric, correct_pattern = 0, 0, 0
        total_train = 0

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total}, Trainable params: {trainable}")

        # Use tqdm to show training progress
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{start_epoch + num_epochs}", leave=False)

        for batch_idx, (images, shape_labels, fabric_labels, pattern_labels) in enumerate(progress_bar):
            images = images.to(device)
            shape_labels = shape_labels.float().to(device)  # Convert to float for BCEWithLogitsLoss
            fabric_labels = fabric_labels.float().to(device)  # Convert to float for BCEWithLogitsLoss
            pattern_labels = pattern_labels.float().to(device)  # Convert to float for BCEWithLogitsLoss

            optimizer.zero_grad()

            # Forward pass
            shape_output, fabric_output, pattern_output = model(images)

            # Compute losses
            loss_shape = shape_criterion(shape_output, shape_labels)
            loss_fabric = fabric_criterion(fabric_output, fabric_labels)
            loss_pattern = pattern_criterion(pattern_output, pattern_labels)

            total_loss = loss_shape + loss_fabric + loss_pattern

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            # Track running loss
            running_loss += total_loss.item()

            # Compute Accuracy (Multi-Label)
            predicted_shape = (torch.sigmoid(shape_output) > 0.5).float()
            predicted_fabric = (torch.sigmoid(fabric_output) > 0.5).float()
            predicted_pattern = (torch.sigmoid(pattern_output) > 0.5).float()

            correct_shape += (predicted_shape == shape_labels).sum().item()
            correct_fabric += (predicted_fabric == fabric_labels).sum().item()
            correct_pattern += (predicted_pattern == pattern_labels).sum().item()

            total_train += images.size(0)  # Batch size

            # Update tqdm progress bar with loss
            progress_bar.set_postfix({"Loss": total_loss.item()})

        # Compute final train loss & accuracy
        train_loss = running_loss / len(train_loader)
        train_shape_acc = correct_shape / (total_train * 50)  # 50 shape attributes
        train_fabric_acc = correct_fabric / (total_train * 24)  # 24 fabric features (3 x 8)
        train_pattern_acc = correct_pattern / (total_train * 24)  # 24 pattern features (3 x 8)

        train_accuracy = (train_shape_acc + train_fabric_acc + train_pattern_acc) / 3  # Average accuracy

        # Validation loss, accuracy, precision and recall
        (val_loss, val_shape_acc, val_fabric_acc, val_pattern_acc,
         val_shape_prec, val_shape_rec, val_fabric_prec, val_fabric_rec,
         val_pattern_prec, val_pattern_rec,
         val_avg_f1, val_shape_f1, val_fabric_f1, val_pattern_f1,
         shape_thresh, fabric_thresh, pattern_thresh) = calculate_loss_and_accuracy(
            val_loader, model, shape_criterion, fabric_criterion, pattern_criterion, device, thresholds, tune=True
        )

        val_accuracy = (val_shape_acc + val_fabric_acc + val_pattern_acc) / 3  # Average accuracy

        print(
            f"📊 Epoch [{epoch + 1}/{start_epoch + num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1-Score: {val_avg_f1:.4f}"
        )
        print(f"Shape Precision: {val_shape_prec:.3f} Recall: {val_shape_rec:.3f} F1-Score: {val_shape_f1:.3f} | "
              f"Fabric Precision: {val_fabric_prec:.3f} Recall: {val_fabric_rec:.3f} F1-Score: {val_fabric_f1:.3f} | "
              f"Pattern Precision: {val_pattern_prec:.3f} Recall: {val_pattern_rec:.3f} F1-Score: {val_pattern_f1:.3f}")

        scheduler.step()

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/Val', val_accuracy, epoch)

        writer.add_scalar('F1/Val', val_avg_f1, epoch)

        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)

        # Save checkpoint
        is_best = val_avg_f1 > best_val_f1
        thresholds = {
            'shape': shape_thresh,
            'fabric': fabric_thresh,
            'pattern': pattern_thresh
        }

        save_checkpoint(epoch, model, optimizer, scheduler, val_accuracy, val_loss, thresholds, is_best)

        # Early stopping
        if is_best:
            best_val_f1 = val_avg_f1
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("⏹️ Early stopping triggered.")
                break

    # Load best model
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    thresholds = checkpoint.get("thresholds", None)

    # Test performance
    (
        test_loss, test_shape_acc, test_fabric_acc, test_pattern_acc,
        test_shape_prec, test_shape_rec, test_fabric_prec, test_fabric_rec,
        test_pattern_prec, test_pattern_rec,
        test_avg_f1, test_shape_f1, test_fabric_f1, test_pattern_f1,
        _, _, _
    ) = calculate_loss_and_accuracy(
        test_loader, model, shape_criterion, fabric_criterion, pattern_criterion, device, thresholds, tune=False
    )

    test_accuracy = (test_shape_acc + test_fabric_acc + test_pattern_acc) / 3

    print(
        f"| Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f} | Test F1-Score: {test_avg_f1}"
    )
    print(f"Shape Precision: {test_shape_prec:.3f} Recall: {test_shape_rec:.3f} F1-Score: {test_shape_f1:.3f}| "
          f"Fabric Precision: {test_fabric_prec:.3f} Recall: {test_fabric_rec:.3f} F1-Score: {test_fabric_f1:.3f}| "
          f"Pattern Precision: {test_pattern_prec:.3f} Recall: {test_pattern_rec:.3f} F1-Score: {test_pattern_f1:.3f}")

    # Log final metrics as hyperparams
    writer.add_hparams({
        'batch_size': batch_size,
        'lr': learning_rate,
        'epochs': num_epochs
    }, {
        'hparam/test_loss': test_loss,
        'hparam/test_accuracy': test_accuracy,
        'hparam/test_f1': test_avg_f1
    })

    writer.close()