
from examples.california_housing import LEARNING_RATE, NUM_EPOCHS
from kanwise import KAN

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

NUM_EPOCHS = 1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GAMMA = 0.8

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
valset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=False)   # Shuffle is False for reproducibility
trainloader2 = DataLoader(trainset, batch_size=64, shuffle=False)  # Shuffle is False for reproducibility
valloader = DataLoader(valset, batch_size=64, shuffle=False)

def train(model, trainloader, optimizer, criterion, scheduler):
    for epoch in range(NUM_EPOCHS):
        model.train()
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, 28 * 28).to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, 28 * 28).to(device)
                output = model(images)
                val_loss += criterion(output, labels.to(device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)

        scheduler.step()

        print(
            f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
        )

model = KAN([28 * 28, 64, 10])
model2 = nn.Sequential(
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
optimizer2 = optim.AdamW(model2.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=GAMMA)


criterion = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()

train(model, trainloader, optimizer, criterion, scheduler)
train(model2, trainloader2, optimizer2, criterion2, scheduler2)

