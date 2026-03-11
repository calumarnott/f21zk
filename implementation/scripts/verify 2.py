from __future__ import annotations
import argparse, json, os
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd

from pendulum_ml.utils import load_cfg, apply_overrides
from pendulum_ml.data.dataset import build_loaders
from pendulum_ml.models.registry import make_model
from pendulum_ml.verification.attacks import make_attack_fn

def load_snapshot(run: str):
    snap = Path("experiments")/run/"config.json"
    if not snap.exists():
        raise SystemExit(f"Missing snapshot: {snap}")
    return json.loads(snap.read_text())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--set", nargs="*")
    args = p.parse_args()

    cfg = load_snapshot(args.run)
    cfg = apply_overrides(cfg, args.set or [])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_loaders(cfg)
    model = make_model(
        cfg["model"]["name"],
        in_dim=int(cfg["model"].get("in_dim", 2)),
        out_dim=int(cfg["model"].get("out_dim", 1)),
        hidden=tuple(cfg["model"].get("hidden", (128,128))),
        dropout=float(cfg["model"].get("dropout", 0.0)),
    ).to(device)
    
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # clean metrics on a slice
    mse = nn.MSELoss(); mae = nn.L1Loss()
    X, Y = next(iter(loaders["test"]))
    X, Y = X.to(device), Y.to(device)
    clean = {"mse": float(mse(model(X), Y).item()), "mae": float(mae(model(X), Y).item())}

    # adversarial
    ver = cfg.get("verify") or {}
    atk = make_attack_fn(
        ver.get("attack", "pgd"),
        ver.get("norm", "linf"),
        float(ver.get("eps", 0.03)),
        ver.get("steps", 40),
        ver.get("alpha", None),
        input_clip=None,
    )
    X_adv = atk(model, X, Y)
    adv = {"mse": float(mse(model(X_adv), Y).item()), "mae": float(mae(model(X_adv), Y).item())}

    out = Path("experiments")/args.run
    out.mkdir(parents=True, exist_ok=True)
    (out/"verification.json").write_text(json.dumps({"clean": clean, "adv": adv}, indent=2))
    pd.DataFrame({
        "y_true": Y.detach().cpu().numpy().ravel(),
        "y_clean": model(X).detach().cpu().numpy().ravel(),
        "y_adv": model(X_adv).detach().cpu().numpy().ravel(),
    }).to_csv(out/"predictions_adv.csv", index=False)

    print({"clean": clean, "adv": adv})
