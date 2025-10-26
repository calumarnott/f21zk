from pathlib import Path
from typing import Optional, Dict, Any, Callable
import json

import torch
import torch.nn as nn
from tqdm import tqdm

from .data.dataset import build_loaders
from .models.registry import make_model


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    split: str = "val",
    save_preds_csv: Optional[Path] = None,
    show_progress: bool = True,
    attack_fn: Optional[Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, Any]:
    """ Evaluate model on a given data loader.

    Args:
        model (torch.nn.Module): model to evaluate
        loader (torch.utils.data.DataLoader): data loader
        device (Optional[torch.device], optional): device to use. Defaults to None.
        split (str, optional): data split name (for logging). Defaults to "val".
        save_preds_csv (Optional[Path], optional): if provided, saves predictions to this CSV file. Defaults to None.
        show_progress (bool, optional): whether to show progress bar. Defaults to True.

    Returns:
        Dict[str, Any]: dictionary with evaluation metrics (mse, mae, n)
    """
    
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    
    # Define loss functions
    mse = nn.MSELoss(reduction="mean")
    mae = nn.L1Loss(reduction="mean")

    total_mse, total_mae, n_total = 0.0, 0.0, 0
    ys, yh = [], [] # for saving predictions if needed

    iterable = tqdm(loader, desc=f"eval[{split}]", leave=False) if show_progress else loader

    for X, y in iterable:
        
        X, y = X.to(device), y.to(device)

        # apply adversarial attack if provided
        if attack_fn is not None:
            # attack_fn returns perturbed X (should be detached, same device)
            X = attack_fn(model, X, y)
        
        y_hat = model(X) # prediction
        
        bs = X.size(0) # batch size
        
        # compute batch metrics
        batch_mse = float(mse(y_hat, y).item())
        batch_mae = float(mae(y_hat, y).item())
        
        # accumulate metrics
        total_mse += float(mse(y_hat, y).item()) * bs
        total_mae += float(mae(y_hat, y).item()) * bs
        n_total += bs
        
        # update bar with batch and running averages
        if show_progress:
            running_mse = total_mse / max(1, n_total)
            running_mae = total_mae / max(1, n_total)
            iterable.set_postfix(
                {
                    "batch_mse": f"{batch_mse:.6f}",
                    "batch_mae": f"{batch_mae:.6f}",
                    "running_mse": f"{running_mse:.6f}",
                    "running_mae": f"{running_mae:.6f}",
                }
            )
        
        # store predictions if needed
        if save_preds_csv is not None:
            ys.append(y.detach().cpu())
            yh.append(y_hat.detach().cpu())

    # final metrics
    results = {
        "split": split,
        "n": n_total,
        "mse": total_mse / max(1, n_total),
        "mae": total_mae / max(1, n_total),
    }

    # save predictions to CSV if needed
    if save_preds_csv is not None and ys and yh:
        import pandas as pd
        
        # concatenate labels and predictions from all batches
        # save 1 column CSV per y_true output and 1 column per y_pred output
        n_outputs = ys[0].shape[1]
        # flatten to dim (N, n_outputs)
        y_true = torch.cat(ys, dim=0).numpy().reshape(-1, n_outputs)
        y_pred = torch.cat(yh, dim=0).numpy().reshape(-1, n_outputs)

        # ensure directory exists
        save_preds_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # save to CSV
        # pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(save_preds_csv, index=False)
        
        df_dict = {}
        for i in range(n_outputs):
            df_dict[f"y_true_{i}"] = y_true[:, i]
            df_dict[f"y_pred_{i}"] = y_pred[:, i]
        pd.DataFrame(df_dict).to_csv(save_preds_csv, index=False)

    return results


@torch.no_grad()
def evaluate_val(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    loaders: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """ Evaluate model on 'val' split.

    Args:
        cfg (Dict[str, Any]): configuration dictionary
        model (torch.nn.Module): model to evaluate
        loaders (Optional[Dict[str, Any]], optional): data loaders. If None, will build from cfg. Defaults to None.
        show_progress (bool, optional): whether to show progress bar. Defaults to False.

    Returns:
        Dict[str, Any]: dictionary with evaluation metrics (mse, mae, n)
    """
    if loaders is None:
        loaders = build_loaders(cfg)
        
    device = next(model.parameters()).device
    
    return evaluate(model, loaders["val"], device=device, split="val", save_preds_csv=None, show_progress=show_progress)


@torch.no_grad()
def evaluate_test(
    cfg: Dict[str, Any],
    ckpt_path: str,
    run: Optional[str] = None,
    loaders: Optional[Dict[str, Any]] = None,
    show_progress: bool = False,
    attack_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """ Evaluate a model checkpoint on the 'test' split.
    If run is provided, saves:
      - experiments/<run>/eval_test.json
      - experiments/<run>/predictions_test.csv

    Args:
        cfg (Dict[str, Any]): configuration dictionary
        ckpt_path (str): path to model checkpoint
        run (Optional[str], optional): run name for saving results. Defaults to None.
        loaders (Optional[Dict[str, Any]], optional): data loaders. If None, will build from cfg. Defaults to None.
        show_progress (bool, optional): whether to show progress bar. Defaults to False.

    Returns:
        Dict[str, Any]: dictionary with evaluation metrics (mse, mae, n)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["device"] = str(device) # update cfg with device info
    
    # load model
    model = make_model(
        cfg["model"]["name"],
        in_dim=int(cfg["model"]["in_dim"]),
        out_dim=int(cfg["model"]["out_dim"]),
        hidden=tuple(cfg["model"]["hidden"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    if loaders is None:
        loaders = build_loaders(cfg)

    # build attack function if requested
    attack_fn = None
    if attack_cfg is None:
        # fallback to cfg["attack"] if present
        attack_cfg = cfg.get("attack", None)

    if attack_cfg:
        # use your attacks factory (adjust import / name as needed)
        from .verification.attacks import make_attack_fn
        attack_fn = make_attack_fn(
            method=str(attack_cfg.get("method", "pgd")),
            norm=str(attack_cfg.get("norm", "linf")),
            eps=float(attack_cfg.get("eps", 0.03)),
            steps=int(attack_cfg.get("steps", 10)) if attack_cfg.get("steps") is not None else None,
            alpha=attack_cfg.get("alpha", None),
            input_clip=None,
        )

    # prepare output directory and prediction CSV path if run is specified
    outdir = Path("experiments") / run if run else None
    preds_csv = (outdir / "predictions_test.csv") if outdir else None
    
    # evaluate on test set
    # res = evaluate(model, loaders["test"], device=device, split="test", save_preds_csv=preds_csv, show_progress=show_progress)
    res = evaluate(model, loaders["test"], device=device, split="test", save_preds_csv=preds_csv, show_progress=show_progress, attack_fn=attack_fn)

    
    # save results to JSON if run is specified
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "eval_test.json").write_text(json.dumps(res, indent=2))

    return res