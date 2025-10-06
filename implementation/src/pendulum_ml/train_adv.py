from __future__ import annotations
from pathlib import Path
import json
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .data.dataset import build_loaders
from .evaluate import evaluate_val, evaluate_test
from .models.registry import make_model
from .verification.attacks import make_attack_fn

def _resolve_device(cfg) -> torch.device:
    default = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(cfg.get("device", default))

def _clip_from_cfg(cfg) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    adv = (cfg.get("train") or {}).get("adv") or {}
    clip = adv.get("clip") or {}
    if not clip.get("enabled", False):
        return None
    lo = torch.tensor(clip.get("lo"), dtype=torch.float32)
    hi = torch.tensor(clip.get("hi"), dtype=torch.float32)
    return lo, hi

def train_adv(cfg):
    """
    Train with optional adversarial training.
    - If cfg.train.adv.enabled == true:
        * Generate adversarial examples on-the-fly per batch (PGD/FGSM).
        * mode="replace": use only adversarial loss
        * mode="mix":     loss = mix_alpha * adv_loss + (1-mix_alpha) * clean_loss
        * p: probability that a batch is attacked (curriculum/compute control)
        * start_epoch: begin adversarial training after N warm-up epochs
    """
    run = cfg.get("exp")
    if not run:
        run = f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path("experiments") / run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    device = _resolve_device(cfg)

    # Data
    loaders = build_loaders(cfg)

    # Model & Optim
    model = make_model(
        cfg["model"]["name"],
        in_dim=int(cfg["model"].get("in_dim", 2)),
        out_dim=int(cfg["model"].get("out_dim", 1)),
        hidden=tuple(cfg["model"].get("hidden", (128, 128))),
        dropout=float(cfg["model"].get("dropout", 0.0)),
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    mse = nn.MSELoss(reduction="mean")

    # Adversarial training setup
    adv_cfg = (cfg.get("train") or {}).get("adv") or {}
    adv_enabled = bool(adv_cfg.get("enabled", False))
    attack_p     = float(adv_cfg.get("p", 1.0))
    adv_mode     = str(adv_cfg.get("mode", "replace")).lower()   # replace|mix
    mix_alpha    = float(adv_cfg.get("mix_alpha", 0.5))
    start_epoch  = int(adv_cfg.get("start_epoch", 0))
    input_clip   = _clip_from_cfg(cfg)
    attack_fn = None
    if adv_enabled:
        attack_fn = make_attack_fn(
            method=str(adv_cfg.get("method", "pgd")),
            norm=str(adv_cfg.get("norm", "linf")),
            eps=float(adv_cfg.get("eps", 0.03)),
            steps=adv_cfg.get("steps", None),
            alpha=adv_cfg.get("alpha", None),
            input_clip=input_clip,
        )

    # Logging
    metrics_csv = (out / "metrics.csv").open("w")
    header = "epoch,train_loss,val_loss"
    if adv_enabled and adv_mode == "mix":
        header += ",train_adv_loss,train_clean_loss"
    metrics_csv.write(header + "\n")

    best_val = float("inf")
    best_ckpt = Path("models/checkpoints") / f"{run}.pt"
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg["train"]["epochs"])

    for epoch in range(epochs):
        model.train()
        tr_sum = 0.0; n_tr = 0
        adv_sum = 0.0; clean_sum = 0.0

        pbar = tqdm(loaders["train"], desc=f"train[{epoch+1}/{epochs}]", leave=False)
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()

            if adv_enabled and epoch >= start_epoch and torch.rand(1).item() <= attack_p:
                # adv batch
                X_adv = attack_fn(model, X, y).to(device)
                y_hat_adv = model(X_adv)
                adv_loss = mse(y_hat_adv, y)
                if adv_mode == "replace":
                    loss = adv_loss
                    clean_loss = None
                elif adv_mode == "mix":
                    y_hat_clean = model(X)
                    clean_loss = mse(y_hat_clean, y)
                    loss = mix_alpha * adv_loss + (1.0 - mix_alpha) * clean_loss
                    clean_sum += float(clean_loss.item()) * X.size(0)
                else:
                    raise ValueError(f"Unknown adv mode: {adv_mode}")
                adv_sum += float(adv_loss.item()) * X.size(0)
            else:
                # clean batch
                y_hat = model(X)
                loss = mse(y_hat, y)
                clean_loss = None

            loss.backward()
            opt.step()

            bs = X.size(0)
            tr_sum += float(loss.item()) * bs
            n_tr   += bs

            # progress bar
            if adv_enabled and adv_mode == "mix" and clean_loss is not None:
                pbar.set_postfix(
                    batch_mse=f"{loss.item():.4f}",
                    avg_mse=f"{tr_sum/max(1,n_tr):.4f}",
                    avg_adv=f"{adv_sum/max(1,n_tr):.4f}",
                    avg_clean=f"{clean_sum/max(1,n_tr):.4f}",
                )
            else:
                pbar.set_postfix(
                    batch_mse=f"{loss.item():.4f}",
                    avg_mse=f"{tr_sum/max(1,n_tr):.4f}",
                )

        train_loss = tr_sum / max(1, n_tr)

        # ---- Validate on CLEAN val split (standard protocol) ----
        val_res = evaluate_val(cfg, model, loaders=loaders, show_progress=False)
        val_loss = float(val_res["mse"])

        # log
        row = f"{epoch},{train_loss:.6f},{val_loss:.6f}"
        if adv_enabled and adv_mode == "mix":
            avg_adv = adv_sum / max(1, n_tr)
            avg_clean = clean_sum / max(1, n_tr)
            row += f",{avg_adv:.6f},{avg_clean:.6f}"
        metrics_csv.write(row + "\n")
        metrics_csv.flush()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_ckpt.as_posix())

    # ---- Final test (CLEAN) ----
    test_res = evaluate_test(cfg, best_ckpt.as_posix(), run=run, loaders=loaders, show_progress=False)

    summary = {
        "best_val": best_val,
        "test_mse": float(test_res["mse"]),
        "test_mae": float(test_res["mae"]),
        "sizes": loaders.get("sizes", {}),
        "ckpt": best_ckpt.as_posix(),
        "device": str(device),
        "adv_enabled": adv_enabled,
        "adv_mode": adv_mode if adv_enabled else None,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    # optional convenience row
    try:
        with (out / "metrics.csv").open("a") as f:
            f.write(f"final,{train_loss:.6f},{best_val:.6f}\n")
    except Exception:
        pass

    return {"run": run, **summary}
