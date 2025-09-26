from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

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
    
    n = len(X) # total number of samples
    n_tr = int(n*cfg["data"]["train_ratio"]) # number of training samples
    n_val = int(n*cfg["data"]["val_ratio"]) # number of validation samples
    n_te = n - n_tr - n_val # number of test samples
    
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
