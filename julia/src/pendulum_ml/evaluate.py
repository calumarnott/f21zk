from pathlib import Path
import json, torch, torch.nn as nn
from .data.dataset import build_loaders
from .models.registry import make_model

def _make_model(cfg):
    return make_model(cfg["model"]["name"],
                      in_dim=2,
                      hidden=tuple(cfg["model"]["hidden"]),
                      out_dim=1,
                      dropout=cfg["model"]["dropout"])

def evaluate(cfg, ckpt_path: str, split: str = "test", run: str | None = None):
    loaders = build_loaders(cfg)
    sizes = loaders["sizes"]

    model = _make_model(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    loss_mse = nn.MSELoss(reduction="mean")
    loss_mae = nn.L1Loss(reduction="mean")

    total_mse, total_mae, n_total = 0.0, 0.0, 0
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for X, y in loaders[split]:
            y_hat = model(X)
            total_mse += loss_mse(y_hat, y).item() * len(X)
            total_mae += loss_mae(y_hat, y).item() * len(X)
            n_total += len(X)
            y_true_all.append(y)
            y_pred_all.append(y_hat)

    import torch as _t
    y_true_all = _t.cat(y_true_all).cpu().numpy()
    y_pred_all = _t.cat(y_pred_all).cpu().numpy()

    mse = total_mse / max(1, n_total)
    mae = total_mae / max(1, n_total)

    # Persist results under experiments/<run>/...
    if run:
        out = Path("experiments")/run
        out.mkdir(parents=True, exist_ok=True)
        (out/f"eval_{split}.json").write_text(json.dumps({"mse": mse, "mae": mae, "n": n_total}, indent=2))
        import pandas as pd
        pd.DataFrame({"y_true": y_true_all.ravel(), "y_pred": y_pred_all.ravel()}).to_csv(out/f"predictions_{split}.csv", index=False)

    return {"split": split, "mse": mse, "mae": mae, "n": n_total, "sizes": sizes}
