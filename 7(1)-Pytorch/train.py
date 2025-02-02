import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Any
from resnet import ResNet, BasicBlock
from config import *
from tqdm import tqdm
import time

NUM_CLASSES = 10  

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # PIL.Image -> Tensor 변환
])

# CIFAR-10 데이터셋 로드
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform = transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform = transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# resnet 18 선언하기
## TODO
model = ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=NUM_CLASSES).to(device)


criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
optimizer: optim.Adam = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습 
def train(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> None:
    model.train()
    total_loss: float = 0
    correct: int = 0
    total: int = 0

    progress_bar = tqdm(loader, desc="Training", leave=True)
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 현재 진행률, Loss, Accuracy 업데이트
        accuracy = 100. * correct / total
        avg_loss = total_loss / (batch_idx + 1)

        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / (batch_idx + 1) * len(loader)
        eta = estimated_total_time - elapsed_time

        progress_bar.set_postfix({
            "Loss": f"{avg_loss:.4f}",
            "Acc": f"{accuracy:.2f}%",
            "ETA": f"{eta:.1f}s"
        })

    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
# def train(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> None:
#     model.train()
#     total_loss: float = 0
#     correct: int = 0
#     total: int = 0

#     for inputs, targets in loader:
#         inputs, targets = inputs.to(device), targets.to(device)

#         outputs = model(inputs)
#         loss = criterion(outputs, targets)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#     accuracy: float = 100. * correct / total
#     print(f"Train Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.2f}%")

# 평가 
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> None:
    model.eval()
    total_loss: float = 0
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy: float = 100. * correct / total
    print(f"Test Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.2f}%")

# 학습 및 평가 루프
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train(model, train_loader, criterion, optimizer, device)
    evaluate(model, test_loader, criterion, device)

# 모델 저장
torch.save(model.state_dict(), "resnet18_checkpoint.pth")
print(f"Model saved to resnet18_checkpoint.pth")
