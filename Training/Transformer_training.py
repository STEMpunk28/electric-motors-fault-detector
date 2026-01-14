from env import *
import csv
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# -------------------------------
# Config
# -------------------------------
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

SAVE_PATH = os.path.join(WEIGHTS_PATH, 'transformer.pth')
CSV_PATH  = os.path.join(WEIGHTS_PATH, 'training_log.csv')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Dataset
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_data = ImageFolder(os.path.join(PATH, 'train'), transform=transform)
val_data   = ImageFolder(os.path.join(PATH, 'val'), transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# ViT feature extractor
# -------------------------------
vit = timm.create_model("vit_tiny_patch16_224", pretrained=True)
vit.reset_classifier(0)   # remove ImageNet head

for p in vit.parameters():
    p.requires_grad = False   # freeze ViT

# -------------------------------
# Full model
# -------------------------------
class SpectrogramViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.vit(x)
        return self.classifier(features)

model = SpectrogramViT().to(DEVICE)

# -------------------------------
# Loss & optimizer
# -------------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR)

# -------------------------------
# CSV Logger
# -------------------------------
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "accuracy", "loss", "val_accuracy", "val_loss"])

# -------------------------------
# Validation function
# -------------------------------
def validate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x).squeeze()
            preds = (torch.sigmoid(logits) > 0.5).long()

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

# -------------------------------
# Training with checkpointing
# -------------------------------
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for x,y in train_loader:
        x = x.to(DEVICE)
        y = y.float().to(DEVICE)

        optimizer.zero_grad()
        logits = model(x).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += y.size(0)

    train_acc = correct / total
    train_loss /= len(train_loader)

    val_acc, val_loss = validate(model, val_loader)

    print(f"Epoch {epoch+1:02d} | acc={train_acc:.4f} loss={train_loss:.4f} | val_acc={val_acc:.4f} val_loss={val_loss:.4f}")

    # Write CSV row
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_acc, train_loss, val_acc, val_loss])

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": best_acc
        }, SAVE_PATH)

        print(f"Best model saved (acc = {best_acc:.4f})")

print("Training complete. Best validation accuracy:", best_acc)
