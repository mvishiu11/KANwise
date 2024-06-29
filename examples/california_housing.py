import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

from kanwise.kan import KAN

BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

def get_data_loader(features, targets, batch_size):
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), 
                             torch.tensor(targets, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train(model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

def main():
    # Load and preprocess data
    data = fetch_california_housing()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape[1]}, Testing data shape: {X_test.shape}")
    
    dataset = {}

    dataset['train_input'] = torch.from_numpy(X_train)
    dataset['test_input'] = torch.from_numpy(X_test)
    dataset['train_label'] = torch.from_numpy(y_train)
    dataset['test_label'] = torch.from_numpy(y_test)

    # Get data loaders
    train_loader = get_data_loader(X_train, y_train, BATCH_SIZE)

    # Setup model
    model = KAN([8, 4, 1]) 
    model2 = SimpleMLP(input_dim=X_train.shape[1], output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion2 = nn.MSELoss()
    optimizer2 = optim.AdamW(model2.parameters(), lr=LEARNING_RATE)

    # Train model
    start_kan = time.time()
    train(model, train_loader, criterion, optimizer)
    end_kan = time.time()   
    start_mlp = time.time()
    train(model2, train_loader, criterion2, optimizer2)
    end_mlp = time.time()

    # Inference
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        predictions = model(test_tensor)
        mse = mean_squared_error(y_test, predictions.numpy())
        print(f"Test MSE: {mse}\nTime for training: {abs(start_kan - end_kan)}")
        
    # Inference for simple MLP
    model2.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        predictions = model2(test_tensor)
        mse = mean_squared_error(y_test, predictions.numpy())
        print(f"Test MSE for simple MLP: {mse}\nTime for training: {abs(start_mlp - end_mlp)}")

if __name__ == "__main__":
    main()