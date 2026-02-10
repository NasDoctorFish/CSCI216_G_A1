# ============================================================
# RAF-DB Emotion Classification with ResNet18
# Single-file clean pipeline
# ============================================================

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Device
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ============================================================
# 2. RAF-DB Loader (RGB, NO normalization here)
# ============================================================
def load_raf_dataset(root_dir, split="train"):
    root_dir = os.path.join(root_dir, split)
    class_names = sorted(os.listdir(root_dir))
    label_map = {name: i for i, name in enumerate(class_names)}

    images, labels = [], []
    for cls in class_names:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            path = os.path.join(cls_dir, fname)
            img = cv2.imread(path)                      # BGR uint8
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
            img = torch.tensor(img, dtype=torch.float32)  # (H,W,3) 0~255

            images.append(img)
            labels.append(label_map[cls])

    return images, torch.tensor(labels), class_names

# ============================================================
# 3. Dataset (resize + normalize ONLY ONCE)
# ============================================================
class RAFDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]            # (H,W,3) 0~255
        # If list, to torch, if numpy to torch
        if isinstance(img, list):
            img = torch.tensor(img)
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        img = img.float()
        img = img.permute(2,0,1)        # (3,H,W)

        img = F.interpolate(
            img.unsqueeze(0),
            size=(224,224),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        img = img / 255.0               # normalize ONCE
        img = (img - self.mean) / self.std

        return img, self.labels[idx]

# ============================================================
# 4. ResNet18 (from scratch)
# ============================================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3,2,1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_ch, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        layers = [BasicBlock(self.in_ch, out_ch, stride, downsample)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x,1))

# ============================================================
# 5. Main
# ============================================================
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_RAF_DIR = os.path.join(SCRIPT_DIR, "..", "RAF_DB", "DATASET")
    BASE_RAF_DIR = os.path.abspath(BASE_RAF_DIR)

    train_imgs, train_labels, class_names = load_raf_dataset(BASE_RAF_DIR, "train")
    test_imgs, test_labels, _ = load_raf_dataset(BASE_RAF_DIR, "test")

    train_dataset = RAFDataset(train_imgs, train_labels)
    test_dataset  = RAFDataset(test_imgs, test_labels)
    
    # print dataset info before DataLoader
    print('\n\n\tBEFORE\n\n')
    for i in range(10):
        print(f'\ntrain_dataset {i}: {train_dataset[i]}')
        print(f'\ntest_dataset {i}: {test_dataset[i]}')
    
    
    

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False,
                              num_workers=0, pin_memory=False)
    
    print('\n\n\tAFTER DataLoader\n\n')
    for imgs, labels in train_loader:
        print(f'\nimgs shape: {imgs.shape}')
        print(f'\nlabels shape: {labels.shape}')

    model = ResNet18(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # This actually trains the model forwarding or optimizing process.

    # ========================================================
    # Training model: epoch 20
    # ========================================================
    EPOCHS = 6

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")


    print("Sanity loss:", loss.item()) # Loss: how wrong is the model's predicion?

    # ========================================================
    # Test Accuracy
    # ========================================================
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100*correct/total:.2f}%")

    # ========================================================
    # Visualization (GT vs PR)
    # ========================================================
    imgs, labels = next(iter(train_loader)) # iteration show first value
    imgs, labels = imgs.to(device), labels.to(device)

    with torch.no_grad():
        preds = model(imgs).argmax(dim=1)

    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(device)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(device)
    imgs_vis = (imgs * std + mean).clamp(0,1)

    plt.figure(figsize=(12,6))
    for i in range(6):
        plt.subplot(2,3,i+1)
        # imgs_vis already used RAF_DATALoader
        img = imgs_vis[i].permute(1,2,0).cpu().numpy()
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"GT: {class_names[labels[i]]}\nPR: {class_names[preds[i]]}")
    plt.tight_layout()
    plt.show()
