from operator import index
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, Dataset

# Add custom dataset class if needed

class TensorDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X, self.Y = X, Y
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]


def build_loaders(cfg):
    """ Build data loaders from processed dataset.

    Args:
        cfg (dict): configuration dictionary

    Returns:
        dict: dictionary containing train, val, test data loaders
    """    
    # Load processed data
    # root = Path("data/processed")/cfg["system"]
    if "processed_tag" in cfg["data"]:
        root = Path(__file__).resolve().parents[3] / "data" / "processed" / cfg["data"]["processed_tag"]
    else:
        root = Path(__file__).resolve().parents[3] / "data" / "processed" / cfg["system"] #makes it independent of where the script or notebook runs.
    
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
    
    # Generate dataset splits
    g = torch.Generator().manual_seed(cfg["seed"]) 
    train_ds, val_ds, test_ds = random_split(full, [r_tr, r_val, r_te], generator=g)
    
    # Data loaders -> they handle batching and shuffling
    mk  = lambda ds, bs, sh: DataLoader(ds, batch_size=bs, shuffle=sh)
    
    bs = cfg["train"]["batch_size"] # batch size
    
    return {"train": mk(train_ds, bs, True), 
            "val": mk(val_ds, bs, False), 
            "test": mk(test_ds, bs, False),
            "sizes": {"train": len(train_ds),
                        "val": len(val_ds),
                        "test": len(test_ds)}}
