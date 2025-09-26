from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

# Add custom dataset class if needed

def build_loaders(cfg):
    """ Build data loaders from processed dataset.

    Args:
        cfg (dict): configuration dictionary

    Returns:
        dict: dictionary containing train, val, test data loaders
    """
    
    # Load processed data
    root = Path("data/processed")/cfg["system"]
    X = torch.load(root/"X.pt")
    y = torch.load(root/"y.pt")
    full = TensorDataset(X, y)
    
    n = len(X) # total number of samples
    r_tr = cfg["data"]["train_ratio"] # training ratio
    r_val = cfg["data"].get("val_ratio", 0.1) # validation ratio
    r_te = cfg["data"].get("test_ratio", 0.1) # test ratio
    
    # Normalize ratios to ensure they sum to 1
    s = r_tr + r_val + r_te
    r_tr, r_val, r_te = r_tr/s, r_val/s, r_te/s
    
    # Determine split sizes
    n_tr = int(round(n * r_tr)) # number of training samples
    n_val = int(round(n * r_val)) # number of validation samples
    n_te = n - n_tr - n_val # number of test samples (remaining) 
    
    g = torch.Generator().manual_seed(cfg["data"]["seed"]) 
    train_ds, val_ds, test_ds = random_split(full, [n_tr, n_val, n_te], generator=g)
    
    # Create datasets
    train_ds = TensorDataset(X[:n_tr], y[:n_tr])
    val_ds   = TensorDataset(X[n_tr:n_tr+n_val], y[n_tr:n_tr+n_val])
    test_ds  = TensorDataset(X[-n_te:], y[-n_te:])
    
    # Data loaders -> they handle batching and shuffling
    mk  = lambda ds, bs, sh: DataLoader(ds, batch_size=bs, shuffle=sh)
    
    bs = cfg["train"]["batch_size"] # batch size
    
    return {"train": mk(train_ds, bs, True), 
            "val": mk(val_ds, bs, False), 
            "test": mk(test_ds, bs, False)}
