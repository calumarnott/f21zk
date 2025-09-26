from pathlib import Path
import json, time, torch, torch.nn as nn, torch.optim as optim
import tqdm
from .data.dataset import build_loaders
from .models.registry import make_model

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
    
   
    loaders = build_loaders(cfg) # build data loaders
    
    # create model from registry
    model = make_model(cfg["model"]["name"],
                        in_dim=2,
                        hidden=tuple(cfg["model"]["hidden"]),
                        out_dim=1,
                        dropout=cfg["model"]["dropout"]) # create model from registry
    
    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["device"] = str(device) # update cfg with device info
    model.to(device)

    # optimizer and loss function
    opt = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss = nn.MSELoss()
    best=float("inf")
    
    # open metrics CSV for logging
    mcsv = (out/"metrics.csv").open("w")
    mcsv.write("epoch,train,val\n") # write header

    # training loop
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0 # training loss
        
        for X,y in tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} train"):
            
            X,y = X.to(device), y.to(device)
            
            opt.zero_grad() # reset gradients
            l = loss(model(X), y) # compute loss
            l.backward() # backpropagate
            opt.step() # update weights
            
            train_loss += l.item() * X.size(0)
            
        train_loss /= len(loaders["train"].dataset) # average training loss accross all samples

        model.eval()
        
        val_loss = 0 # validation loss
        
        with torch.no_grad():
            
            for X,y in tqdm(loaders["val"], desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} val"):
                
                X,y = X.to(device), y.to(device)
                
                val_loss += loss(model(X), y).item() * X.size(0)
                
        val_loss /= len(loaders["val"].dataset)

        # log metrics
        mcsv.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")
        mcsv.flush() # ensure data is written to disk
        
        # save best model
        if val_loss < best: 
            
            best = val_loss
            
            # setup checkpoint path
            ckpt = Path("models/checkpoints")/f"{run}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            
            # save model state
            torch.save(model.state_dict(), ckpt.as_posix())

    # log best validation loss to metrics.json
    (out/"metrics.json").write_text(json.dumps({"best_val": best}, indent=2))
    
    # return run id and best validation loss
    return {"run": run, 
            "best_val": best}
