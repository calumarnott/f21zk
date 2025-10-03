from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

# --------- helpers ---------
def _project(delta: torch.Tensor, eps: float, norm: str) -> torch.Tensor:
    if norm == "linf":
        return delta.clamp_(-eps, eps)
    if norm == "l2":
        flat = delta.view(delta.size(0), -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        factors = (eps / norms).clamp_max(1.0)
        flat = flat * factors
        return flat.view_as(delta)
    raise ValueError(f"Unknown norm: {norm}")

def _clip_inputs(x: torch.Tensor, clip: Optional[Tuple[torch.Tensor, torch.Tensor]]):
    if clip is None: return x
    lo, hi = clip
    # Broadcast if needed
    if not torch.is_tensor(lo): lo = torch.tensor(lo, dtype=x.dtype, device=x.device)
    if not torch.is_tensor(hi): hi = torch.tensor(hi, dtype=x.dtype, device=x.device)
    return x.clamp(lo, hi)

# --------- FGSM (single-step) for regression (MSE loss) ---------
@torch.enable_grad()
def fgsm_regression(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float,
    norm: str = "linf",
    input_clip: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    model.eval()
    x0 = x.detach().clone().requires_grad_(True)
    y_hat = model(x0)
    loss = nn.functional.mse_loss(y_hat, y, reduction="mean")
    grad, = torch.autograd.grad(loss, x0)
    with torch.no_grad():
        if norm == "linf":
            x_adv = x0 + eps * grad.sign()
        elif norm == "l2":
            g = grad.view(grad.size(0), -1)
            g = g / (g.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))
            x_adv = x0 + eps * g.view_as(x0)
        else:
            raise ValueError(f"Unknown norm: {norm}")
        x_adv = _clip_inputs(x_adv, input_clip)
    return x_adv.detach()

# --------- PGD (multi-step) for regression (MSE loss) ---------
@torch.enable_grad()
def pgd_regression(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float,
    steps: int,
    alpha: Optional[float] = None,
    norm: str = "linf",
    input_clip: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    random_start: bool = True,
) -> torch.Tensor:
    model.eval()
    x0 = x.detach()
    if alpha is None:
        alpha = eps / max(1, steps)

    # init
    if random_start:
        if norm == "linf":
            delta = torch.empty_like(x0).uniform_(-eps, eps)
        elif norm == "l2":
            noise = torch.randn_like(x0)
            noise = noise / (noise.view(noise.size(0), -1).norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))
            delta = noise * eps
        else:
            raise ValueError(f"Unknown norm: {norm}")
    else:
        delta = torch.zeros_like(x0)

    delta.requires_grad_(True)

    for _ in range(steps):
        x_adv = x0 + delta
        x_adv = _clip_inputs(x_adv, input_clip)
        y_hat = model(x_adv)
        loss = nn.functional.mse_loss(y_hat, y, reduction="mean")
        grad, = torch.autograd.grad(loss, delta, retain_graph=False)
        with torch.no_grad():
            if norm == "linf":
                delta += alpha * grad.sign()
            elif norm == "l2":
                g = grad.view(grad.size(0), -1)
                g = g / (g.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))
                delta += alpha * g.view_as(delta)
            delta = _project(delta, eps, norm)
    x_adv = x0 + delta.detach()
    x_adv = _clip_inputs(x_adv, input_clip)
    return x_adv.detach()

# --------- factory ---------
def make_attack_fn(method: str, norm: str, eps: float, steps: int | None, alpha: float | None,
                   input_clip: Optional[Tuple[torch.Tensor, torch.Tensor]]):
    method = method.lower()
    norm = norm.lower()
    if method == "fgsm":
        return lambda model, X, Y: fgsm_regression(model, X, Y, eps=eps, norm=norm, input_clip=input_clip)
    if method == "pgd":
        s = int(steps or 10)
        a = alpha
        return lambda model, X, Y: pgd_regression(model, X, Y, eps=eps, steps=s, alpha=a, norm=norm, input_clip=input_clip)
    raise ValueError(f"Unknown attack method: {method}")
