import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from models import SimpleNN
from dataset import PendulumDataset

# Hyperparameters
hidden_dim = 256
output_dim = 1  # e.g., predicting next theta_dot
learning_rate = 0.001
batch_size = 32
num_epochs = 30



def make_splits(dataset, train_frac=0.8, seed=42):
    n = len(dataset)
    n_tr = int(n * train_frac)
    n_te = n - n_tr
    g = torch.Generator().manual_seed(seed)
    train_subset, test_subset = random_split(dataset, [n_tr, n_te], generator=g)

    # Compute normalization from TRAIN ONLY
    Xtr = dataset.X[train_subset.indices]
    ytr = dataset.y[train_subset.indices]
    Xte = dataset.X[test_subset.indices]
    yte = dataset.y[test_subset.indices]

    mu = Xtr.mean(dim=0, keepdim=True)
    sd = Xtr.std(dim=0, keepdim=True).clamp_min(1e-8)

    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd

    train_ds = TensorDataset(Xtr_n, ytr)
    test_ds  = TensorDataset(Xte_n, yte)
    return train_ds, test_ds

def train():
    full_ds = PendulumDataset('data/pendulum_simulation.csv')

    # i.i.d. random split (as before)
    train_ds, test_ds = make_splits(full_ds, train_frac=0.8, seed=42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = SimpleNN(input_dim=train_ds.tensors[0].shape[1],
                     hidden_dim=hidden_dim,
                     output_dim=output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        train_running = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_running += loss.item() * Xb.size(0)
        train_loss = train_running / len(train_ds)

        # ---- Test ----
        model.eval()
        test_running = 0.0
        with torch.no_grad():
            for Xb, yb in test_loader:
                pred = model(Xb)
                test_running += criterion(pred, yb).item() * Xb.size(0)
        test_loss = test_running / len(test_ds)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1:02d} | train MSE: {train_loss:.6f} | test MSE: {test_loss:.6f}")

    # ---- Plot losses ----
    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train MSE")
    plt.plot(epochs, test_losses, label="Test MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training and Test Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    # Comment out if running headless:
    plt.show()

    # Save model
    torch.save(model.state_dict(), "data/pendulum_model.pth")

if __name__ == "__main__":
    train()