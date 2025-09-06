import torch
import torch.nn as nn
from torch.nn.functional import relu, cross_entropy
from torchvision.datasets import ImageFolder
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import functional as TF
from PIL import Image
from torch import Tensor

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,  16, kernel_size=3, stride=1, padding=1), # (16,28,28)
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), # (16,14,14)
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32,14,14)
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), # (32,7,7)
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc  = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) # logits
        return x

EPOCH_COUNT = 100

def train_small_cnn(cnn: SmallCNN):
    device = next(cnn.parameters()).device
    lr = 0.01
    optimizer = SGD(cnn.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH_COUNT)
    best_acc = 0
    for epoch in range(EPOCH_COUNT):
        print(f"Starting epoch {epoch}")
        cnn.train()
        avg_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # forward
            z = cnn(images)
            
            # loss
            loss = cross_entropy(z, labels)
            avg_loss += loss.item()
            
            # backward
            optimizer.zero_grad()
            loss.backward()

            # update
            optimizer.step()
        
        scheduler.step() # Once per epoch, not per batch like an idiot would do (not me)!!
        
        # eval
        acc = evaluate_small_cnn(cnn, val_loader)
        print(f"Finished epoch {epoch}, avg_loss: {avg_loss/len(train_loader):.4f}, val_acc: {acc}, LR: {scheduler.get_lr()[0]:.4f}")
        if acc > best_acc:
            best_acc = acc
            print(f"Found new best model with acc: {acc}")
            torch.save(cnn.state_dict(), "model.pth")

def evaluate_small_cnn(cnn: SmallCNN, loader: DataLoader):
    cnn.eval()
    device = next(cnn.parameters()).device
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            z = cnn(images)
            preds = torch.argmax(z, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

def alpha_as_input(img: Tensor) -> Tensor:
    return img[3:4,:,:]

transform = transforms.Compose([
    transforms.ToTensor(),        # converts HxW or HxWxC → tensor
    transforms.Lambda(alpha_as_input),   # invert colors
    # transforms.Normalize(0.5, 0.5)  # optional: scale to [-1,1]
])

def rgba_loader(path):
    img = Image.open(path).convert("RGBA")
    return img
image_folder = ImageFolder(Path("dataset"), transform=transform, loader=rgba_loader)
print(f"Num classes: {len(image_folder.classes)}")

gen = torch.Generator().manual_seed(42)

train_size, val_size, test_size = 0.8, 0.1, 0.1
subset_size = 5000
image_folder_reduced, _ = random_split(image_folder, [subset_size, len(image_folder)-subset_size], gen)
[train_dataset, val_dataset, test_dataset] = random_split(
    image_folder_reduced, 
    (train_size, val_size, test_size),
    generator=gen
)
print(f"Sizes: train_dataset={len(train_dataset)} val_dataset={len(val_dataset)}, test_dataset={len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

imgs, labels = next(iter(train_loader))
print(imgs.min().item(), imgs.max().item())

if __name__ == "__main__":
    print("hello")
    print(str(torch.__version__))

    cnn = SmallCNN(num_classes=10)
    cnn.to("cpu")
    train_small_cnn(cnn)


    