import torch
import torch.nn as nn
from torch.nn.functional import relu, cross_entropy
from torchvision.datasets import ImageFolder
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,  16, kernel_size=3, stride=1, padding=1) # (16,28,28)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1) # (16,14,14)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0) # (32,14,14)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1) # (32,7,7)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc  = nn.Linear(32, 128)
    
    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) # logits
        return x

def train_small_cnn(cnn: SmallCNN):
    device = next(cnn.parameters()).device
    step = 0.01
    best_acc = 0
    for epoch in range(100):
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
            cnn.zero_grad()
            loss.backward()

            # update
            for param in cnn.parameters():
                param.data += -step * param.grad
        
        # eval
        acc = evaluate_small_cnn(cnn, val_loader)
        print(f"Finished epoch {epoch}, avg_loss: {avg_loss/(epoch+1)}, val_acc: {acc}")
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
            preds = torch.argmax(z, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

transform = transforms.Compose([
    transforms.Grayscale(),       # ensure single channel if needed
    transforms.ToTensor(),        # converts HxW or HxWxC → tensor
    transforms.Normalize(0.5, 0.5)  # optional: scale to [-1,1]
])

image_folder = ImageFolder(Path("dataset"), transform=transform)

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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


if __name__ == "__main__":
    print("hello")
    print(str(torch.__version__))

    cnn = SmallCNN()
    cnn.to("cuda")
    train_small_cnn(cnn)


    