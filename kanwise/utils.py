import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def load_data(path):
    """
    Load data from a CSV file.
    :param path: str, path to the CSV file.
    :return: DataFrame, the loaded data.
    """
    return pd.read_csv(path)


class CustomDataset(Dataset):
    def __init__(self, data, feature_cols, target_col):
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(
            self.data.iloc[idx][self.feature_cols].values.astype(float),
            dtype=torch.float32,
        )
        y = torch.tensor(self.data.iloc[idx][self.target_col], dtype=torch.float32)
        return x, y


def get_dataloader(filepath, batch_size, feature_cols, target_col):
    data = pd.read_csv(filepath)
    dataset = CustomDataset(data, feature_cols, target_col)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
