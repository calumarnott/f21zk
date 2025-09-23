from torch.utils.data import Dataset
import pandas as pd
import torch


# Custom Dataset
class PendulumDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.X = torch.tensor(data.iloc[:, 1:-1].values, dtype=torch.float32)
        print(f"Input features shape: {self.X.shape}")
        self.y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]