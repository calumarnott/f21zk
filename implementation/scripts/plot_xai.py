#!/usr/bin/env python3

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["KMP_WARNINGS"] = "FALSE"
os.environ["MKL_VERBOSE"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

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
# p.add_argument("--alpha", type=float, default=0.005)
p.add_argument("--verbose", action="store_true",)
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

# If full header exists (features + outputs)
if len(all_names) >= in_dim + out_dim:
    feature_names = all_names[:in_dim]
    output_names = all_names[-out_dim:]
else:
    # Fallback if config doesn’t include all names
    feature_names = all_names[:in_dim] or [f"f{i}" for i in range(in_dim)]
    output_names = [f"out{i}" for i in range(out_dim)]

print(f"[INFO] Using {len(feature_names)} input features: {feature_names}")
print(f"[INFO] Using {len(output_names)} outputs: {output_names}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = make_model(
    cfg["model"]["name"],
    in_dim=in_dim,
    hidden=tuple(cfg["model"].get("hidden", (128, 128))),
    out_dim=out_dim,
    dropout=float(cfg["model"].get("dropout", 0.0)),
).to(device)

ckpt = run_dir / "model_best.pth"
if ckpt.exists():
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"[INFO] Loaded weights from {ckpt}")
else:
    print(f"[WARN] No checkpoint found at {ckpt}")

model.eval()

# =====================================================
# 3. Load test data
# =====================================================
loaders = build_loaders(cfg)
test_loader = loaders["test"]

X_list, Y_list = [], []
for X_batch, Y_batch in test_loader:
    X_list.append(X_batch)
    Y_list.append(Y_batch)
X = torch.cat(X_list, dim=0).to(device)
Y = torch.cat(Y_list, dim=0).to(device)

print(f"[INFO] Loaded real test set: X={X.shape}, Y={Y.shape}")

# =====================================================
# 4. Attack setup
# =====================================================
alpha=args.eps/max(1, args.steps)

attack_fn = make_attack_fn(
    method=args.attack,
    eps=args.eps,
    steps=args.steps,
    alpha=alpha,
    norm="linf",
)
print(f"[INFO] Using attack: {args.attack.upper()} (eps={args.eps}, steps={args.steps}, alpha={alpha})")

# =====================================================
# 5. Plotting functions
# =====================================================

def plot_lime_comparison(model, X, attack_fn, idx=42):
    """
    Compare LIME explanations (clean vs adversarial) for each output.
    """
    x = X[idx:idx+1]
    with torch.no_grad():
        y = model(x)
    x_adv = attack_fn(model, x, y)

    # Create explainer using first 100 samples as reference
    explainer = xai.explain_lime(model, X[:100].detach().cpu(), feature_names)

    out_dim = y.shape[1] if y.ndim > 1 else 1

    for j in range(out_dim):
        exp_clean = xai.lime_explain_instance(model, explainer, x[0], output_idx=j, verbose=args.verbose)
        exp_adv = xai.lime_explain_instance(model, explainer, x_adv[0], output_idx=j, verbose=False)
        drift = xai.attribution_drift(exp_clean, exp_adv)

        clean_dict = dict(exp_clean)
        adv_dict = dict(exp_adv)
        weights_clean = [clean_dict.get(f, 0.0) for f in feature_names]
        weights_adv = [adv_dict.get(f, 0.0) for f in feature_names]

        out_name = output_names[j] if j < len(output_names) else f"out{j}"

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].bar(feature_names, weights_clean, color="cornflowerblue")
        ax[1].bar(feature_names, weights_adv, color="indianred")

        for a in ax:
            a.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
            a.set_ylabel("LIME weight")

        ax[0].set_title(f"Clean {out_name}")
        ax[1].set_title(f"Adversarial {out_name}")
        fig.suptitle(f"LIME drift ({out_name}) = {drift:.3f}", fontsize=12)
        plt.tight_layout()

        out_path = figs / f"lime_comparison_{out_name}_{args.attack}.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[INFO] Saved: {out_path}")


def summarize_lime_drift(model, X, attack_fn, n_samples=30):
    """
    Compute and plot LIME drift distribution across random test samples.
    """
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
    """
    Plot Integrated Gradients for each output dimension of one example.
    """
    x = X[idx:idx+1].requires_grad_(True)
    with torch.no_grad():
        y = model(x)
    out_dim = y.shape[1] if y.ndim > 1 else 1

    for j in range(out_dim):
        attrs, _ = xai.integrated_gradients(model, x, target=j)
        plt.figure(figsize=(4, 3))
        plt.bar(feature_names, attrs.detach().cpu().numpy().flatten(), color="seagreen")
        plt.ylabel("Attribution")
        plt.title(f"Integrated Gradients (Output {j})")
        plt.tight_layout()
        out_path = figs / f"integrated_gradients_output{j}_{args.attack}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[INFO] Saved: {out_path}")

# potential feature entaglemnet plot
def plot_feature_entanglement(model, X, feature_names, n_samples=100):
    """
    using Integrated Gradients correlation heatmap.
    """
    model.eval()
    X_sub = X[:n_samples].requires_grad_(True)
    attrs_all = []

    print(f"[INFO] Computing IG attributions for {n_samples} samples...")

    for i in range(n_samples):
        x = X_sub[i:i+1]
        with torch.no_grad():
            y = model(x)
        out_dim = y.shape[1] if y.ndim > 1 else 1

        # Compute IG for each output and average (for multi-output models)
        attr_accum = torch.zeros_like(x)
        for j in range(out_dim):
            attrs, _ = xai.integrated_gradients(model, x, target=j)
            attr_accum += attrs.abs()  # absolute attribution magnitude
        attrs_all.append(attr_accum.squeeze(0).cpu().numpy())

    attrs_all = np.stack(attrs_all, axis=0)  # shape (n_samples, n_features)

    # Compute correlation between feature attributions
    corr = np.corrcoef(attrs_all.T)

    # === Plot ===
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr, xticklabels=feature_names, yticklabels=feature_names,
        cmap="coolwarm", center=0, square=True, cbar_kws={"label": "Correlation"}
    )
    plt.title("Feature Entanglement (IG Correlation Heatmap)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    out_path = figs / "feature_entanglement_IG_corr.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved: {out_path}")


# =====================================================
# 6. Run
# =====================================================
plot_lime_comparison(model, X, attack_fn)
plot_integrated_gradients(model, X)
summarize_lime_drift(model, X, attack_fn)
plot_feature_entanglement(model, X, feature_names)

print(f"✅ Explainability plots saved in: {figs}")
