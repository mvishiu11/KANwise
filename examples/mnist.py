import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from kanwise import KAN

# Argument parser setup
parser = argparse.ArgumentParser(description="Train KAN and MLP models on MNIST")
parser.add_argument(
    "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
)
args = parser.parse_args()

NUM_EPOCHS = args.epochs
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
trainloader = DataLoader(
    trainset, batch_size=64, shuffle=False  # Shuffle is False for reproducibility
)
trainloader2 = DataLoader(
    trainset, batch_size=64, shuffle=False  # Shuffle is False for reproducibility
)
valloader = DataLoader(valset, batch_size=64, shuffle=False)


def train(model, trainloader, optimizer, criterion, scheduler):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss, epoch_accuracy = 0, 0
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, 28 * 28).to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                pbar.set_postfix(
                    loss=loss.item(),
                    accuracy=accuracy.item(),
                    lr=optimizer.param_groups[0]["lr"],
                )

        epoch_loss /= len(trainloader)
        epoch_accuracy /= len(trainloader)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        model.eval()
        val_loss, val_accuracy = 0, 0
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
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step()

        print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

    return train_losses, train_accuracies, val_losses, val_accuracies


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
model2.to(device)
optimizer2 = optim.AdamW(
    model2.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=GAMMA)

criterion = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()

train_losses_kan, train_accuracies_kan, val_losses_kan, val_accuracies_kan = train(
    model, trainloader, optimizer, criterion, scheduler
)
train_losses_mlp, train_accuracies_mlp, val_losses_mlp, val_accuracies_mlp = train(
    model2, trainloader2, optimizer2, criterion2, scheduler2
)

# Plotting
epochs = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses_kan, label="KAN Train Loss")
plt.plot(epochs, val_losses_kan, label="KAN Val Loss")
plt.plot(epochs, train_losses_mlp, label="MLP Train Loss")
plt.plot(epochs, val_losses_mlp, label="MLP Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies_kan, label="KAN Train Accuracy")
plt.plot(epochs, val_accuracies_kan, label="KAN Val Accuracy")
plt.plot(epochs, train_accuracies_mlp, label="MLP Train Accuracy")
plt.plot(epochs, val_accuracies_mlp, label="MLP Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")

plt.tight_layout()
plt.show()
