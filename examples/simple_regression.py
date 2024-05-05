# examples/simple_regression.py
import os

import torch
import torch.optim as optim

from kanwise.config import BATCH_SIZE
from kanwise.config import LEARNING_RATE
from kanwise.config import NUM_EPOCHS
from kanwise.models import KANModel
from kanwise.utils import get_dataloader


def train(model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(NUM_EPOCHS):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


def main():
    data_path = os.path.join("data", "synthetic_data.csv")
    feature_cols = ["Feature"]
    target_col = "Target"
    train_loader = get_dataloader(data_path, BATCH_SIZE, feature_cols, target_col)
    model = KANModel()
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, train_loader, criterion, optimizer)


if __name__ == "__main__":
    main()
