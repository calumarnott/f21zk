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

from pendulum_ml.data.dataset import build_loaders
from pendulum_ml.models.registry import make_model
from pendulum_ml.verification.attacks import make_attack_fn
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

# === Feature & Output naming ===

all_names = cfg["model"].get("features", [])
in_dim = int(cfg["model"].get("in_dim", len(all_names)))
out_dim = int(cfg["model"].get("out_dim", 1))

# Split feature and output names automatically
if len(all_names) >= in_dim + out_dim:
    feature_names = all_names[:in_dim]
    output_names = all_names[-out_dim:]
else:
    # Fallback if config only had inputs
    feature_names = all_names[:in_dim] or [f"f{i}" for i in range(in_dim)]
    output_names = [f"out{i}" for i in range(out_dim)]

print(f"[INFO] Using {len(feature_names)} input features: {feature_names}")
print(f"[INFO] Using {len(output_names)} output names: {output_names}")

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
feature_names = cfg["model"].get("features", ["theta", "omega", "length"])

# =====================================================
# 3. Load test data
# =====================================================
loaders = build_loaders(cfg)
test_loader = loaders["test"]

# Collect all test samples into a single tensor (or sample a subset)
X_list, Y_list = [], []
for X_batch, Y_batch in test_loader:
    X_list.append(X_batch)
    Y_list.append(Y_batch)
X = torch.cat(X_list, dim=0).to(device)
Y = torch.cat(Y_list, dim=0).to(device)

print(f"[INFO] Loaded real test set: X={X.shape}, Y={Y.shape}")

# =====================================================
# 4. Attack
# =====================================================
attack_fn = make_attack_fn(
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
    """Compare LIME explanations for each output (clean vs adversarial)."""
    x = X[idx:idx+1]
    with torch.no_grad():
        y = model(x)
    x_adv = attack_fn(model, x, y)

    # Create LIME explainer using reference samples
    explainer = xai.explain_lime(model, X[:100].detach().cpu(), feature_names)

    out_dim = y.shape[1] if y.ndim > 1 else 1

    for j in range(out_dim):
        # Explain a specific output dimension j
        exp_clean = xai.lime_explain_instance(model, explainer, x[0], output_idx=j)
        exp_adv = xai.lime_explain_instance(model, explainer, x_adv[0], output_idx=j)
        drift = xai.attribution_drift(exp_clean, exp_adv)

        # Convert to dictionaries for consistent alignment
        clean_dict = dict(exp_clean)
        adv_dict = dict(exp_adv)

        # Ensure all features are represented (missing -> 0 weight)
        weights_clean = [clean_dict.get(f, 0.0) for f in feature_names]
        weights_adv = [adv_dict.get(f, 0.0) for f in feature_names]

        # Identify output name (from config-derived list)
        out_name = output_names[j] if j < len(output_names) else f"out{j}"

        # Plot comparison
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].bar(feature_names, weights_clean, color="cornflowerblue")
        ax[1].bar(feature_names, weights_adv, color="indianred")

        # Formatting
        for a in ax:
            a.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
            a.set_ylabel("LIME weight")

        ax[0].set_title(f"LIME (Clean – {out_name})")
        ax[1].set_title(f"LIME (Adversarial – {out_name})")
        fig.suptitle(f"LIME drift ({out_name}) = {drift:.3f}", fontsize=12)
        plt.tight_layout()

        # Save figure
        out_path = figs / f"lime_comparison_{out_name}_{args.attack}.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[INFO] Saved: {out_path}")


def summarize_lime_drift(model, X, attack_fn, n_samples=30):
    explainer = xai.explain_lime(model, X[:100].detach().cpu(), feature_names)
    with torch.no_grad():
        y = model(X[:1])
    out_dim = y.shape[1] if y.ndim > 1 else 1

    for j in range(out_dim):
        idxs = np.random.choice(len(X), n_samples, replace=False)
        drifts = []
        for i in idxs:
            x = X[i:i+1]
            with torch.no_grad():
                y = model(x)
            x_adv = attack_fn(model, x, y)
            exp_clean = xai.lime_explain_instance(model, explainer, x[0], output_idx=j)
            exp_adv = xai.lime_explain_instance(model, explainer, x_adv[0], output_idx=j)
            drifts.append(xai.attribution_drift(exp_clean, exp_adv))
        drifts = np.array(drifts)
        mean_d, std_d = drifts.mean(), drifts.std()

        plt.figure(figsize=(5, 3))
        plt.hist(drifts, bins=10, color="orchid", edgecolor="k", alpha=0.8)
        plt.xlabel("LIME drift (L1)")
        plt.ylabel("Count")
        plt.title(f"LIME Drift (output {j}, {args.attack.upper()})")
        plt.tight_layout()
        out_path = figs / f"lime_drift_output{j}_{args.attack}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[RESULT] Output {j}: Mean LIME drift {mean_d:.4f} ± {std_d:.4f}")


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
