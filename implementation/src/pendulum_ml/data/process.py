from pathlib import Path
import json, torch, pandas as pd
from typing import Optional, Union
import numpy as np
from ..utils import import_system


def processed_dir_for(cfg) -> Path:
    tag = cfg.get("data", {}).get("processed_tag") or cfg.get("system")
    return Path("data/processed") / str(tag)

# Helper: add repetitions of initial state to fill first window
def first_window(x0: Union[torch.Tensor, np.ndarray], y0: Optional[Union[torch.Tensor, np.ndarray]], length: int):
    """ Create the first window by repeating the initial state.
    
    Args:
        x0 (torch.Tensor): initial state, shape [C]
        y0 (torch.Tensor): initial label, shape [D]
        length (int): length of the window
    """
    # Ensure x0 is a torch tensor
    if isinstance(x0, np.ndarray):
        x0 = torch.from_numpy(x0).to(dtype=torch.float32)
    elif not isinstance(x0, torch.Tensor):
        x0 = torch.tensor(x0, dtype=torch.float32)

    # Repeat to create window of shape (length, C)
    window = x0.unsqueeze(0).repeat(length, 1)  # shape (length, C)

    # Normalize/convert label if provided
    label = None
    if y0 is not None:
        if isinstance(y0, np.ndarray):
            label = torch.from_numpy(y0).to(dtype=torch.float32)
        elif isinstance(y0, torch.Tensor):
            label = y0
        else:
            label = torch.tensor(y0, dtype=torch.float32)

    return window, label

def make_windows(X_seq: torch.Tensor, y_seq: torch.Tensor,
                 length: int, stride: int):
    """ Create sliding windows from sequential data.

    Args:
        X_seq (torch.Tensor): sequence of feature vectors, shape [T, C]
        y_seq (torch.Tensor): sequence of target vectors, shape [T, D]
        length (int): length of each window
        stride (int): stride between windows

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - windows tensor of shape [N, C, length]
            - labels tensor of shape [N, D]
    """
    
    # X_seq: [T, C] -> Time steps, C channels
    # y_seq: [T, D] -> Time steps, D labels
    T, C = X_seq.shape
    
    assert length > 0 and stride > 0, "Window length and stride must be positive integers."
    assert length <= T, "Window length must be less than or equal to the sequence length."
    
    windows, labels = [], []
    

    last_start = T - length

    if last_start < 0:
        return None, None
    
    # first window
    win, lab = first_window(X_seq[0], y_seq[0], length)
    windows.append(win.T)  # transpose to [C, length]
    labels.append(lab)
    
    for t0 in range(0, last_start + 1, stride):
        t1 = t0 + length
        label_t = (t1 - 1)
        label_t = min(label_t, T - 1)
        
        windows.append(X_seq[t0:t1].T)
        labels.append(y_seq[label_t])

    # [N, C, W] -> N windows, C channels, W window length
    Xw = torch.stack(windows, dim=0)
    # [N, D]    -> N windows, D labels
    Yw = torch.stack(labels, dim=0)

    return Xw, Yw

    

def to_processed(raw_path, cfg, out_dir=None):
    """ Convert raw CSV trajectory data to processed tensors and save normalization stats.

    Args:
        raw_path (str or Path): Path to raw CSV file.
        cfg (dict): Configuration dictionary.
        out_dir (str, optional): Output directory to save processed data. Defaults to "data/processed/pendulum".

    Returns:
        dict: Dictionary containing paths to saved tensors "X" and "y".
    """
    out = Path(out_dir) if out_dir is not None else processed_dir_for(cfg)
    out.mkdir(parents=True, exist_ok=True) # ensure output directory exists
    
    # load raw CSV data
    df = pd.read_csv(raw_path)
    
    cps = import_system(cfg["system"])
    
    # if using windows
    W = int(cfg["data"].get("window", {}).get("length", 0))
    S = int(cfg["data"].get("window", {}).get("stride", 0))
    use_windows = (W > 0 and S > 0)
    
    
    # load features and labels from dataframe as tensors
    
    X_seq = torch.tensor(df[cps.STATE_NAMES + [f"error_{axis}" for axis in cps.INPUT_ERROR_AXES]].values, dtype=torch.float32)
    y_seq = torch.tensor(df[[f"u_{axis}" for axis in cps.OUTPUT_CONTROL_AXES]].values, dtype=torch.float32)

    if use_windows:
        X, y = make_windows(X_seq, y_seq, W, S)
        
        mu = X.mean((0,2), keepdim=True)  # mean over N and T, per channel
        sd = X.std((0,2), keepdim=True).clamp_min(1e-8)  # if sd is too small, clamp to avoid division by zero
        
    else:
        # features:
        X, y = X_seq, y_seq
        
        # normalize features
        mu = X.mean(0, keepdim=True) # mean over T, per channel
        sd = X.std(0, keepdim=True).clamp_min(1e-8) # if sd is too small, clamp to avoid division by zero
    
    # standardize (zero mean, unit variance)
    if cfg["data"].get("standardize", True):
        X = (X - mu)/sd
        
    # save tensors
    torch.save(X, out/"X.pt")
    torch.save(y, out/"y.pt")
    
    # save normalization stats
    (out/"norms.json").write_text(json.dumps({"mu": mu.tolist(), 
                                              "sd": sd.tolist()}))
    
    # return paths to saved tensors
    return {"X": str(out/"X.pt"), 
            "y": str(out/"y.pt")}
