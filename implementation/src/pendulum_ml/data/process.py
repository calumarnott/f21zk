from pathlib import Path
import json, torch, pandas as pd

def to_processed(raw_paths, cfg, out_dir="data/processed/pendulum"):
    """ Convert raw CSV trajectory data to processed tensors and save normalization stats.

    Args:
        raw_paths (list): List of paths to raw CSV files.
        cfg (dict): Configuration dictionary.
        out_dir (str, optional): Output directory to save processed data. Defaults to "data/processed/pendulum".

    Returns:
        dict: Dictionary containing paths to saved tensors "X" and "y".
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True) # ensure output directory exists
    
    df = pd.concat([pd.read_csv(p) for p in raw_paths], ignore_index=True) # combine all CSVs into one DataFrame
    
    # features: theta, theta_dot; target: torque
    X = torch.tensor(df[["theta","theta_dot"]].values, dtype=torch.float32)
    y = torch.tensor(df[["torque"]].values, dtype=torch.float32)

    # normalize features
    mu = X.mean(0, keepdim=True)
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
