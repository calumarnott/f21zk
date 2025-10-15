# verification/xai.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from captum.attr import IntegratedGradients, Saliency, NoiseTunnel

def explain_lime(model, X_ref, feature_names, num_features=None, mode="regression"):
    """Initialize LIME explainer for a regression model."""
    X_np = X_ref.detach().cpu().numpy()
    return LimeTabularExplainer(
        X_np, feature_names=feature_names, mode=mode,
        discretize_continuous=False, verbose=False
    )

def lime_explain_instance(model, explainer, x, num_features=2):
    def model_predict(X_numpy):
        X_t = torch.tensor(X_numpy, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(X_t).cpu().numpy()
        return y_pred
    exp = explainer.explain_instance(x.cpu().numpy(), model_predict, num_features=num_features)
    return exp.as_list()

def integrated_gradients(model, x, baseline=None, steps=50):
    ig = IntegratedGradients(model)
    if baseline is None:
        baseline = torch.zeros_like(x)
    attrs, delta = ig.attribute(x, baseline, n_steps=steps, return_convergence_delta=True)
    return attrs.detach(), delta

def saliency_map(model, x):
    sal = Saliency(model)
    grads = sal.attribute(x)
    return grads.detach()

def smoothgrad(model, x, stdev=0.02, n_samples=25):
    sal = Saliency(model)
    nt = NoiseTunnel(sal)
    smg = nt.attribute(x, nt_type='smoothgrad', stdevs=stdev, nt_samples=n_samples)
    return smg.detach()

def attribution_drift(lime_before, lime_after):
    """Compute L1 drift between two LIME weight lists."""
    d = 0.0
    for (f1, w1), (f2, w2) in zip(lime_before, lime_after):
        d += abs(w1 - w2)
    return d

def plot_feature_attributions(attrs, feature_names):
    """Simple bar plot for gradient-based attributions."""
    vals = attrs.cpu().numpy().flatten()
    plt.bar(feature_names, vals)
    plt.ylabel("Attribution value")
    plt.show()
