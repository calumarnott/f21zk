#!/usr/bin/env python3
"""
plot_xai.py
Visualize explainability (LIME / IG / drift) for a trained controller.

Usage:
    python plot_xai.py --run <run_id> [--attack pgd]
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pendulum_ml.models.registry import make_model
from pendulum_ml.verification.attack import make_attack
from pendulum_ml.verification import xai


# =====================================================
# 1. Command line arguments
# =====================================================
p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="<run> id (in experiments/<run>)")
p.add_argument("--attack", default="pgd", help="Attack method (pgd, fgsm, etc.)")
p.add_argument("--eps", type=float, default=0.02)
p.add_argument("--steps", type=int, default=10)
p.add_argument("--alpha", type=float, default=0.005)
args = p.parse_args()

run_dir = Path("experiments") / args.run
if not run_dir.exists():
    raise SystemExit(f"Missing run directory: {run_dir}")

# Output folder for XAI figures
figs = run_dir / "figures" / "xai"
figs.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. Load config + model
# =====================================================
config_path = run_dir / "config.json"
if not config_path.exists():
    raise SystemExit(f"Missing config: {config_path}")
cfg = json.loads(config_path.read_text())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = make_model(
    cfg["model"]["name"],
    in_dim=int(cfg["model"].get("in_dim", 2)),
    hidden=tuple(cfg["model"].get("hidden", (128, 128))),
    out_dim=int(cfg["model"].get("out_dim", 1)),
    dropout=float(cfg["model"].get("dropout", 0.0)),
).to(device)

ckpt = run_dir / "model_best.pth"
if ckpt.exists():
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"[INFO] Loaded weights from {ckpt}")
else:
    print(f"[WARN] No checkpoint found at {ckpt}")

model.eval()
feature_names = cfg["model"].get("features", ["theta", "omega"])

# =====================================================
# 3. Test data
# =====================================================
def make_test_data(n=200):
    theta = np.linspace(-0.3, 0.3, n)
    omega = np.linspace(-1.0, 1.0, n)
    X = np.stack([theta, omega], axis=1)
    return torch.tensor(X, dtype=torch.float32).to(device)

X = make_test_data()

# =====================================================
# 4. Attack
# =====================================================
attack_fn = make_attack(
    method=args.attack,
    eps=args.eps,
    steps=args.steps,
    alpha=args.alpha,
    norm="linf",
)
print(f"[INFO] Using attack: {args.attack.upper()} (eps={args.eps})")

# =====================================================
# 5. Plotting functions
# =====================================================
def plot_lime_comparison(model, X, attack_fn, idx=42):
    """Compare LIME explanations for one sample (clean vs adversarial)."""
    x = X[idx:idx+1]
    with torch.no_grad():
        y = model(x)
    x_adv = attack_fn(model, x, y)

    explainer = xai.explain_lime(model, X[:100].detach().cpu(), feature_names)
    exp_clean = xai.lime_explain_instance(model, explainer, x[0])
    exp_adv = xai.lime_explain_instance(model, explainer, x_adv[0])
    drift = xai.attribution_drift(exp_clean, exp_adv)

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    labels, w_clean = zip(*exp_clean)
    _, w_adv = zip(*exp_adv)
    ax[0].bar(labels, w_clean, color="cornflowerblue")
    ax[1].bar(labels, w_adv, color="indianred")
    ax[0].set_title("LIME (Clean)")
    ax[1].set_title("LIME (Adversarial)")
    fig.suptitle(f"LIME drift = {drift:.3f}", fontsize=12)
    plt.tight_layout()
    out_path = figs / f"lime_comparison_{args.attack}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved: {out_path}")

def summarize_lime_drift(model, X, attack_fn, n_samples=30):
    """Compute average LIME drift across samples."""
    explainer = xai.explain_lime(model, X[:100].detach().cpu(), feature_names)
    idxs = np.random.choice(len(X), n_samples, replace=False)
    drifts = []
    for i in idxs:
        x = X[i:i+1]
        with torch.no_grad():
            y = model(x)
        x_adv = attack_fn(model, x, y)
        exp_clean = xai.lime_explain_instance(model, explainer, x[0])
        exp_adv = xai.lime_explain_instance(model, explainer, x_adv[0])
        drifts.append(xai.attribution_drift(exp_clean, exp_adv))
    drifts = np.array(drifts)
    mean_d, std_d = drifts.mean(), drifts.std()
    plt.figure(figsize=(5, 3))
    plt.hist(drifts, bins=10, color="orchid", edgecolor="k", alpha=0.8)
    plt.xlabel("LIME drift (L1)")
    plt.ylabel("Count")
    plt.title(f"LIME Drift ({args.attack.upper()})")
    plt.tight_layout()
    out_path = figs / f"lime_drift_{args.attack}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[RESULT] Mean LIME drift: {mean_d:.4f} ± {std_d:.4f}")
    print(f"[INFO] Saved: {out_path}")

def plot_integrated_gradients(model, X, idx=42):
    """Integrated Gradients for one example."""
    x = X[idx:idx+1].requires_grad_(True)
    attrs, _ = xai.integrated_gradients(model, x)
    plt.figure(figsize=(4, 3))
    plt.bar(feature_names, attrs.detach().cpu().numpy().flatten(), color="seagreen")
    plt.ylabel("Attribution")
    plt.title("Integrated Gradients")
    plt.tight_layout()
    out_path = figs / f"integrated_gradients_{args.attack}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved: {out_path}")

# =====================================================
# 6. Run
# =====================================================
plot_lime_comparison(model, X, attack_fn)
plot_integrated_gradients(model, X)
summarize_lime_drift(model, X, attack_fn)

print(f" Explainability plots saved in: {figs}")
