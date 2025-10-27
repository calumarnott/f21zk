# robustness.py
# Minimal helpers for Lipschitz-style robustness and per-feature sensitivity.
# PyTorch + matplotlib only. No seaborn.

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math
import torch
import matplotlib.pyplot as plt


# -----------------------------
# Internal utilities
# -----------------------------

def _flatten2(x: torch.Tensor) -> torch.Tensor:
    """Flatten everything except batch dimension."""
    return x.flatten(1)


def _as_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)


def _take_first_batch(
    loader,
    device: torch.device,
    max_batch_size: Optional[int] = None,
):
    batch = next(iter(loader))
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    x = _as_device(x, device)
    if max_batch_size is not None:
        x = x[:max_batch_size]
    return x


# -----------------------------
# Lipschitz metrics
# -----------------------------

@torch.no_grad()
def empirical_local_lipschitz(
    model: torch.nn.Module,
    x: torch.Tensor,
    radius: float = 0.05,
    trials: int = 20,
) -> float:
    """
    Estimate local Lipschitz constant around x with random L2-bounded perturbations.
    Returns mean over batch.
    """
    model.eval()
    y0 = model(x)
    best = torch.zeros(x.size(0), device=x.device)
    for _ in range(trials):
        u = torch.randn_like(x) # random perturbation of same shape as x
        u_norm = _flatten2(u).norm(p=2, dim=1).clamp_min(1e-12) # u norm
        u = u * (radius / u_norm).view(-1, *([1] * (x.ndim - 1))) # scale to radius
        y = model(x + u) # compute perturbed output
        num = _flatten2(y - y0).norm(p=2, dim=1) # output change norm
        den = _flatten2(u).norm(p=2, dim=1).clamp_min(1e-12) # input change norm
        best = torch.maximum(best, num / den) # update highest found ratio
    return best.mean().item()


def jacobian_frobenius_norm(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> float:
    """
    Average Frobenius norm of the Jacobian ||J_f(x)||_F over the batch.
    Interpretable as a local Lipschitz upper bound (Frobenius >= spectral).
    """
    model.eval()
    x = x.detach().requires_grad_(True)
    y = model(x)
    if y.ndim == 1:
        y = y.unsqueeze(1)

    total = 0.0
    for i in range(y.shape[1]): # for each output dimension
        
        # compute gradient of output i w.r.t. input x
        grad_i = torch.autograd.grad(
            y[:, i].sum(), x, create_graph=False, retain_graph=True
        )[0]
        total += _flatten2(grad_i).pow(2).sum(dim=1)  # squared sum of gradients ||∂y_i/∂x||_F^2
    frob = (total / y.shape[1]).sqrt().mean().item() # average over outputs and batch
    return float(frob)


@torch.no_grad()
def spectral_norm_upper_bound(
    model: torch.nn.Module,
    iters: int = 15,
) -> float:
    """
    Quick global upper bound: product of spectral norms of Linear layers
    (assuming 1-Lipschitz activations like ReLU for a rough bound).
    """
    def power_iter(W: torch.Tensor, iters: int) -> float:
        v = torch.randn(W.shape[1], device=W.device)
        v = v / (v.norm() + 1e-12)
        u = None
        for _ in range(iters):
            u = (W @ v);  u = u / (u.norm() + 1e-12)
            v = (W.T @ u); v = v / (v.norm() + 1e-12)
        return float((u @ (W @ v)).abs().item())

    K = 1.0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            K *= power_iter(m.weight.detach(), iters)
    return float(K)


# -----------------------------
# Per-feature sensitivity (Jacobian entries)
# -----------------------------

def feature_sensitivity(
    model: torch.nn.Module,
    x: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Returns |∂y_i/∂x_j| aggregated over the batch.
    Shape: [n_outputs, n_inputs(flattened)].

    reduce: "mean" or "median" across batch.
    """
    assert reduce in {"mean", "median"}
    model.eval()
    x = x.detach().requires_grad_(True)
    y = model(x)
    if y.ndim == 1:
        y = y.unsqueeze(1)

    B, n_out, n_in = x.shape[0], y.shape[1], x[0].numel()
    sens = torch.zeros(n_out, n_in, device=x.device)

    grads_per_i = []
    for i in range(n_out):
        grad_i = torch.autograd.grad(y[:, i].sum(), x, retain_graph=True)[0]
        grads_per_i.append(grad_i.abs().flatten(1))  # [B, n_in]

    # Stack over outputs -> [n_out, B, n_in]
    G = torch.stack(grads_per_i, dim=0)

    if reduce == "mean":
        sens = G.mean(dim=1)  # [n_out, n_in]
    else:
        sens = G.median(dim=1).values

    return sens.detach().cpu()


# -----------------------------
# Plotting helpers (matplotlib only)
# -----------------------------

def plot_feature_sensitivity_heatmap(
    S: torch.Tensor,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    input_shape: Optional[Tuple[int, ...]] = None,
    figsize: Tuple[float, float] = (8, 4),
    title: str = "Per-feature sensitivity (|∂y_i/∂x_j|)",
):
    """
    Heatmap (outputs x inputs). If input_shape is provided for 1D signals,
    we reshape columns back to (T,) and show a separate line plot per output instead.
    """
    import numpy as np

    S_np = S.numpy()
    n_out, n_in = S_np.shape

    # If a 1D structure exists, offer line plots per output (clearer for sequences).
    if input_shape is not None and len(input_shape) == 1 and math.prod(input_shape) == n_in:
        T = input_shape[0]
        xs = np.arange(T)
        fig, axes = plt.subplots(n_out, 1, figsize=(figsize[0], max(figsize[1], 2*n_out)), sharex=True)
        if n_out == 1:
            axes = [axes]
        for i in range(n_out):
            axes[i].plot(xs, S_np[i].reshape(T))
            axes[i].set_ylabel(f"out {i}" if not output_names else output_names[i])
        axes[-1].set_xlabel("input index" if input_names is None else " • ".join(input_names))
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
        return

    # Fallback: standard heatmap (imshow)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(S_np, aspect='auto')
    ax.set_xlabel("Input features")
    ax.set_ylabel("Output dimensions")
    if input_names is not None:
        ax.set_xticks(range(len(input_names)))
        ax.set_xticklabels(input_names, rotation=90)
    if output_names is not None:
        ax.set_yticks(range(len(output_names)))
        ax.set_yticklabels(output_names)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("|∂y/∂x|")
    plt.tight_layout()
    plt.show()


def plot_per_input_bar(
    S: torch.Tensor,
    aggregate: str = "mean",
    top_k: Optional[int] = None,
    input_names: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (9, 3.5),
    title: str = "Average sensitivity per input feature",
):
    """
    Bar plot over inputs by aggregating across outputs.
    aggregate: "mean" or "max" over outputs.
    top_k: if set, show only top-k most sensitive inputs.
    """
    import numpy as np

    assert aggregate in {"mean", "max"}
    S_np = S.numpy()
    if aggregate == "mean":
        per_input = S_np.mean(axis=0)
    else:
        per_input = S_np.max(axis=0)

    idx = np.arange(len(per_input))
    if top_k is not None and top_k < len(per_input):
        top_idx = np.argsort(-per_input)[:top_k]
        per_input = per_input[top_idx]
        idx = top_idx

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(per_input)), per_input)
    if input_names is None:
        ax.set_xlabel("Input index")
        ax.set_xticks([])
    else:
        chosen_names = [input_names[i] for i in idx]
        ax.set_xticks(range(len(chosen_names)))
        ax.set_xticklabels(chosen_names, rotation=90)
        ax.set_xlabel("Input feature")
    ax.set_ylabel("Aggregated |∂y/∂x|")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# -----------------------------
# High-level convenience
# -----------------------------

def summarize_lipschitz(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    radii: Sequence[float] = (0.01, 0.05, 0.1),
    trials: int = 20,
    jacobian_batch: int = 32,
    print_header: Optional[str] = None,
    include_spectral_bound: bool = False,
) -> Dict[str, float]:
    """
    Prints and returns a compact summary dict:
      - empirical local Lipschitz for a few radii
      - Jacobian Frobenius norm (local upper bound)
      - spectral-norm product upper bound (global rough bound) - optional
    """
    model.eval()
    x = _take_first_batch(loader, device, max_batch_size=max(32, jacobian_batch))

    results = {}
    if print_header:
        print(f"\n=== {print_header} ===")

    # empirical L - compute once per radius
    for r in radii:
        l_r = empirical_local_lipschitz(model, x, radius=r, trials=trials)
        results[f"empirical_L_r={r}"] = l_r
        print(f"Empirical local L @ radius={r:.3f}: {l_r:.6f}")

    # jacobian Frobenius (use smaller sub-batch to be safe)
    jac_x = x[:jacobian_batch]
    jac = jacobian_frobenius_norm(model, jac_x)
    results["jacobian_frobenius"] = jac
    print(f"Jacobian Frobenius norm (local UB): {jac:.6f}")

    # spectral upper bound - only if requested (often too loose to be useful)
    if include_spectral_bound:
        spec = spectral_norm_upper_bound(model)
        results["spectral_product_upper_bound"] = spec
        print(f"Spectral product upper bound (global, rough): {spec:.6f}")

    return results


def compare_models_summary(
    models: Dict[str, torch.nn.Module],
    loader,
    device: torch.device,
    radii: Sequence[float] = (0.01, 0.05, 0.1),
    trials: int = 20,
    jacobian_batch: int = 32,
    include_spectral_bound: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Runs summarize_lipschitz over multiple named models, prints side-by-side blocks,
    and returns a nested dict of results.
    """
    out = {}
    for name, mdl in models.items():
        out[name] = summarize_lipschitz(
            mdl, loader, device,
            radii=radii, trials=trials, jacobian_batch=jacobian_batch,
            print_header=name, include_spectral_bound=include_spectral_bound,
        )
    return out


def compute_and_plot_feature_sensitivity(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batch_size: int = 32,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    input_shape: Optional[Tuple[int, ...]] = None,
    top_k_bar: Optional[int] = None,
):
    """
    Convenience wrapper:
      - grabs a small batch
      - computes |∂y_i/∂x_j| matrix
      - shows a heatmap (or line plots if 1D)
      - shows a per-input bar chart (aggregated over outputs)
    """
    x = _take_first_batch(loader, device, max_batch_size=max_batch_size)
    S = feature_sensitivity(model, x, reduce="mean")
    plot_feature_sensitivity_heatmap(
        S,
        input_names=input_names,
        output_names=output_names,
        input_shape=input_shape,
        title="Per-feature sensitivity (|∂y_i/∂x_j|, averaged over batch)",
    )
    plot_per_input_bar(
        S,
        aggregate="mean",
        top_k=top_k_bar,
        input_names=input_names,
        title="Average sensitivity per input feature (mean over outputs)",
    )
    return S  # return in case the caller wants to reuse/save it
