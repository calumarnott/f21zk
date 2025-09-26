from pathlib import Path
import json, time, torch, torch.nn as nn, torch.optim as optim
import tqdm
from .data.dataset import build_loaders
from .models.registry import make_model
from .evaluate import evaluate_test, evaluate_val

def train(cfg):
    """ Train an MLP model based on the provided configuration.

    Args:
        cfg (dict): Configuration dictionary containing training parameters.

    Returns:
        dict: Dictionary containing the run identifier and best validation loss.
    """
    
    run = cfg.get("exp", f'run-{time.strftime("%Y%m%d-%H%M%S")}') # unique run id
    
    # create experiment directory
    out = Path("experiments")/run
    out.mkdir(parents=True, exist_ok=True)
    
    # save config
    (out/"config.json").write_text(json.dumps(cfg, indent=2))
    
    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["device"] = str(device) # update cfg with device info
    
    loaders = build_loaders(cfg) # build data loaders
    
    # create model from registry
    model = make_model(cfg["model"]["name"],
                        in_dim=2,
                        hidden=tuple(cfg["model"]["hidden"]),
                        out_dim=1,
                        dropout=cfg["model"]["dropout"]).to(device) # create model from registry

    # optimizer and loss function
    opt = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    criterion = nn.MSELoss()
    
    # Logging
    metrics_csv = (out / "metrics.csv").open("w")
    metrics_csv.write("epoch,train_loss,val_loss\n")
    
    best_val = float("inf")
    best_ckpt = Path("models/checkpoints") / f"{run}.pt"
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)

    epochs = cfg["train"]["epochs"]

    # training loop
    for epoch in range(epochs):
        # ---- Train epoch ----
        model.train()
        
        train_loss = 0 # training loss
        
        for X, y in tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{epochs} train"):
            
            X,y = X.to(device), y.to(device)
            
            opt.zero_grad() # reset gradients
            loss = criterion(model(X), y) # compute loss
            loss.backward() # backpropagate
            opt.step() # update weights
            
            train_loss += loss.item() * X.size(0)
            
        train_loss /= len(loaders["train"].dataset) # average training loss accross all samples

        # ---- Validate (in memory, no checkpoint I/O) ----
        val_res = evaluate_val(cfg, model, loaders=loaders, show_progress=False)
        val_loss = float(val_res["mse"])

        # log epoch
        metrics_csv.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")
        metrics_csv.flush()
        
        # keep best by val MSE
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_ckpt.as_posix())

    # ---- Final test on best ckpt (writes JSON + predictions CSV) ----
    test_res = evaluate_test(cfg, best_ckpt.as_posix(), run=run, loaders=loaders, show_progress=False)

    # Summary JSON (kept small and readable)
    summary = {
        "best_val": best_val,
        "test_mse": float(test_res["mse"]),
        "test_mae": float(test_res["mae"]),
        "sizes": loaders.get("sizes", {}),
        "ckpt": best_ckpt.as_posix(),
        "device": str(device),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    
    # return run id and best validation loss
    return {"run": run, 
            **summary} # '**' unpacks the summary dictionary
