from __future__ import annotations
from typing import Optional, Tuple, Callable
import torch
import torch.nn as nn
import math

# -------------------- helpers ------------------------
def _project(delta: torch.Tensor, eps: float, norm: str) -> torch.Tensor:
    norm = norm.lower()
    if norm == "linf":
        return delta.clamp(-eps, eps)
    if norm == "l2":
        flat = delta.view(delta.size(0), -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        factors = (eps / norms).clamp_max(1.0)
        flat = flat * factors
        return flat.view_as(delta)
    raise ValueError(f"Unknown norm: {norm}")

def _clip_inputs(x: torch.Tensor, clip: Optional[Tuple[torch.Tensor, torch.Tensor]]):
    if clip is None:
        return x
    lo, hi = clip
    if not torch.is_tensor(lo):
        lo = torch.tensor(lo, dtype=x.dtype, device=x.device)
    if not torch.is_tensor(hi):
        hi = torch.tensor(hi, dtype=x.dtype, device=x.device)
    return x.clamp(lo, hi)

def _normalize_l2(v: torch.Tensor) -> torch.Tensor:
    flat = v.view(v.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / norms).view_as(v)

# -------------------- FGSM(1 Step) ---------------------------

@torch.enable_grad()
def fgsm_regression(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *,
                    eps: float, norm: str="linf",
                    input_clip: Optional[Tuple[torch.Tensor, torch.Tensor]]=None) -> torch.Tensor:
    was_training = model.training
    model.eval()
    x0 = x.detach().clone().requires_grad_(True)
    y_hat = model(x0)
    loss = nn.functional.mse_loss(y_hat, y, reduction="mean")
    grad, = torch.autograd.grad(loss, x0)
    with torch.no_grad():
        if norm == "linf":
            x_adv = x0 + eps * grad.sign()
        elif norm == "l2":
            x_adv = x0 + eps * _normalize_l2(grad)
        else:
            raise ValueError(f"Unknown norm: {norm}")
        x_adv = _clip_inputs(x_adv, input_clip)
    model.train(was_training)
    return x_adv.detach()


# -------------------- PGD / BIM ----------------------


def pgd_regression(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, *,
    eps: float, steps: int, alpha: Optional[float] = None,
    norm: str = "linf", input_clip=None,
    random_start: bool = True, momentum: float = 0.0
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    x0 = x.detach()

    if alpha is None:
        alpha = eps / max(1, steps)

    # init delta
    if random_start:
        if norm == "linf":
            delta = torch.empty_like(x0).uniform_(-eps, eps)
        else:  # l2
            delta = eps * _normalize_l2(torch.randn_like(x0))
    else:
        delta = torch.zeros_like(x0)

    g_m = torch.zeros_like(delta) if momentum > 0 else None

    for _ in range(steps):
        # make sure delta is a leaf that requires grad each step
        delta = delta.detach().requires_grad_(True)

        with torch.enable_grad():
            x_adv = _clip_inputs(x0 + delta, input_clip)
            pred  = model(x_adv)
            loss  = nn.functional.mse_loss(pred, y, reduction="mean")

            (grad,) = torch.autograd.grad(
                loss, delta, retain_graph=False, create_graph=False
            )

        # PGD update (gradient ascent on loss) without tracking grads
        with torch.no_grad():
            if momentum and g_m is not None:
                g_m = momentum * g_m + grad / (grad.abs().mean() + 1e-12)
                grad_use = g_m
            else:
                grad_use = grad

            if norm == "linf":
                delta = delta + alpha * grad_use.sign()
            else:  # l2
                delta = delta + alpha * _normalize_l2(grad_use)

            delta = _project(delta, eps, norm)

            # keep within valid input range if provided
            if input_clip is not None:
                delta = (_clip_inputs(x0 + delta, input_clip) - x0)

    x_adv = _clip_inputs(x0 + delta.detach(), input_clip)
    model.train(was_training)
    return x_adv



# -------------------- R+FGSM -------------------------
# (random init then one FGSM step)

def rfgsm_regression(model, x, y, *, eps: float, alpha: float,
                     norm: str="linf", input_clip=None):
    x_rand = x + torch.empty_like(x).uniform_(-eps, eps)
    return fgsm_regression(model, x_rand, y, eps=alpha, norm=norm, input_clip=input_clip)


# -------------------- DeepFool (minimal-norm) --------
# For regression: move until output error sign flips

@torch.enable_grad()
def deepfool_regression(model, x, y, *, max_iter=50, overshoot=0.02,
                        eps=None, input_clip=None):
    """
    Approximate minimal L2 perturbation that changes sign of residual.
    Mostly illustrative; not as stable for large nets.
    """
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    for _ in range(max_iter):
        y_hat = model(x_adv)
        loss = nn.functional.mse_loss(y_hat, y, reduction="mean")
        grad, = torch.autograd.grad(loss, x_adv)
        with torch.no_grad():
            step = loss / (grad.view(grad.size(0), -1).norm(p=2, dim=1, keepdim=True) + 1e-12)
            x_adv = x_adv + (1 + overshoot) * step.view_as(x_adv) * _normalize_l2(grad)
            if eps is not None:
                x_adv = x + _project(x_adv - x, eps, "l2")
            x_adv = _clip_inputs(x_adv, input_clip)
    return x_adv.detach()


# -------------------- SPSA (black-box) ---------------

def spsa_regression(model, x, y, *, eps: float, steps: int,
                    alpha: float, sigma: float=1e-3,
                    samples_per_step: int=64, norm: str="linf",
                    input_clip=None):
    model_fn = lambda z: model(z).detach()
    x0 = x.detach()
    delta = torch.zeros_like(x0)
    for _ in range(steps):
        g_est = torch.zeros_like(x0)
        for _s in range(samples_per_step // 2):
            v = torch.randint_like(x0, 0, 2).float() * 2 - 1
            x_plus = _clip_inputs(x0 + delta + sigma * v, input_clip)
            x_minus = _clip_inputs(x0 + delta - sigma * v, input_clip)
            with torch.no_grad():
                loss_p = nn.functional.mse_loss(model_fn(x_plus), y, reduction="mean")
                loss_m = nn.functional.mse_loss(model_fn(x_minus), y, reduction="mean")
            g_est += ((loss_p - loss_m) / (2 * sigma)) * v
        g_est /= float(samples_per_step // 2)
        with torch.no_grad():
            if norm == "linf":
                delta += alpha * g_est.sign()
                delta = delta.clamp(-eps, eps)
            else:
                delta += alpha * _normalize_l2(g_est)
                delta = _project(delta, eps, "l2")
            delta = torch.clamp(delta, -eps, eps)
    return _clip_inputs(x0 + delta, input_clip)


# -------------------- factory ------------------------

def make_attack_fn(method: str, *, eps: float, norm: str="linf",
                steps: int|None=None, alpha: float|None=None,
                input_clip=None, random_start=True,
                momentum: float=0.0, **kw) -> Callable:
    method = method.lower()
    norm = norm.lower()
    if method == "fgsm":
        return lambda m,X,Y: fgsm_regression(m,X,Y,eps=eps,norm=norm,input_clip=input_clip)
    if method in {"pgd","bim"}:
        s = int(steps or 10)
        a = alpha or eps/2
        return lambda m,X,Y: pgd_regression(m,X,Y,eps=eps,steps=s,alpha=alpha,
                                            norm=norm,input_clip=input_clip,
                                            random_start=(method!="bim"),
                                            momentum=momentum)
    if method == "rfgsm":
        a = alpha or eps/2
        return lambda m,X,Y: rfgsm_regression(m,X,Y,eps=eps,alpha=a,norm=norm,input_clip=input_clip)
    if method == "mifgsm":
        s = int(steps or 10)
        return lambda m,X,Y: pgd_regression(m,X,Y,eps=eps,steps=s,alpha=alpha,
                                            norm=norm,input_clip=input_clip,
                                            random_start=True,momentum=1.0)
    if method == "deepfool":
        return lambda m,X,Y: deepfool_regression(m,X,Y,eps=eps,input_clip=input_clip)
    if method == "spsa":
        s = int(steps or 10)
        a = alpha or eps/float(s)
        return lambda m,X,Y: spsa_regression(m,X,Y,eps=eps,steps=s,alpha=a,norm=norm,input_clip=input_clip)
    raise ValueError(f"Unknown attack: {method}")
