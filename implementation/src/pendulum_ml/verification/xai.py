import sys, os
sys.stderr = open(os.devnull, 'w')

import torch
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from captum.attr import IntegratedGradients, Saliency, NoiseTunnel


# ======================================================
# 1. LIME EXPLAINERS
# ======================================================

def explain_lime(model, X_ref, feature_names, num_features=None, mode="regression"):
    """
    Initialize a LIME explainer for a regression model.

    Args:
        model: torch model (not used directly here)
        X_ref: reference samples (tensor) to estimate feature distribution
        feature_names: list of input feature names
        num_features: optional, number of top features to include in explanations
        mode: 'regression' or 'classification'
    """
    X_np = X_ref.detach().cpu().numpy()
    return LimeTabularExplainer(
        X_np,
        feature_names=feature_names,
        mode=mode,
        discretize_continuous=False,
        verbose=False
    )


def lime_explain_instance(model, explainer, x, num_features=5, output_idx=None, verbose=False):
    """
    Explain a single input sample (x) for one output dimension.

    Args:
        model: torch model
        explainer: LimeTabularExplainer object
        x: single input tensor (1D)
        num_features: number of features to display
        output_idx: which output dimension to explain (for multi-output models)
        verbose: if True, prints Boston-style diagnostic info
    """

    # ---- Model wrapper for LIME ----
    def model_predict(X_numpy):
        X_t = torch.tensor(X_numpy, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(X_t).cpu().numpy()  # shape (N, out_dim)
        # Handle multi-output regression
        if y_pred.ndim == 2:
            out_dim = y_pred.shape[1]
            if output_idx is None:
                j = 0
            else:
                j = min(output_idx, out_dim - 1)
            y_pred = y_pred[:, j]
        return y_pred

    # ---- Generate local surrogate explanation ----
    exp = explainer.explain_instance(
        x.detach().cpu().numpy(),
        model_predict,
        num_features=num_features
    )

    if verbose:
        try:
            pred_true = model_predict(x.cpu().numpy().reshape(1, -1))[0]
            local_pred = exp.local_pred[0]
            intercept = exp.intercept[0]
            print(
                f"[LIME: out={output_idx}] "
                f"Intercept={intercept:.4f}, Local={local_pred:.4f}, True={pred_true:.4f}"
            )
        except Exception as e:
            print(f"[LIME diag skipped: {e}]")

    # ---- Map indices safely to feature names ----
    try:
        explanation = exp.as_list()
    except IndexError:
        # Safe manual remap if LIME indices go out of bounds
        used_feats = exp.local_exp.get(1, exp.local_exp.get(0, []))
        n_feats = len(explainer.feature_names)
        explanation = []
        for idx, weight in used_feats:
            if 0 <= idx < n_feats:
                explanation.append((explainer.feature_names[idx], weight))

    return explanation


# 2. GRADIENT-BASED ATTRIBUTIONS


def integrated_gradients(model, x, baseline=None, steps=50, target=None):
    """
    Integrated Gradients attribution for regression model.

    Args:
        model: torch model
        x: input tensor
        baseline: optional baseline tensor (defaults to zeros)
        steps: number of integration steps
        target: index of output neuron to attribute (required for multi-output)
    """
    ig = IntegratedGradients(model)
    if baseline is None:
        baseline = torch.zeros_like(x)
    attrs, delta = ig.attribute(
        x, baseline, target=target, n_steps=steps, return_convergence_delta=True
    )
    return attrs.detach(), delta



def saliency_map(model, x):
    """
    basic saliency map: gradients of output wrt input.
    """
    sal = Saliency(model)
    grads = sal.attribute(x)
    return grads.detach()


def smoothgrad(model, x, stdev=0.02, n_samples=25):
    """
    SmoothGrad: average gradients over noisy samples.
    """
    sal = Saliency(model)
    nt = NoiseTunnel(sal)
    smg = nt.attribute(x, nt_type='smoothgrad', stdevs=stdev, nt_samples=n_samples)
    return smg.detach()



# 3. LIME STABILITY & PLOTTING

def attribution_drift(lime_before, lime_after):
    """
    Compute L1 drift between two LIME explanations (matched by feature name).
    """
    before_dict = dict(lime_before)
    after_dict = dict(lime_after)
    all_feats = set(before_dict.keys()).union(after_dict.keys())
    drift = sum(abs(before_dict.get(f, 0.0) - after_dict.get(f, 0.0)) for f in all_feats)
    return drift


def plot_feature_attributions(attrs, feature_names):
    """
    Simple bar plot for gradient-based attributions.
    """
    vals = attrs.cpu().numpy().flatten()
    plt.bar(feature_names, vals)
    plt.ylabel("Attribution value")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
