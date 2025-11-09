from __future__ import annotations
from .mlp import MLP
from .cnn import CNN1D
from inspect import signature
from typing import Dict, Any, Optional
# add other models here as needed

REGISTRY = {
  "mlp": MLP,
  "cnn": CNN1D,
  # add other models here as needed
}

def _filter_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """ Filter kwargs to only those accepted by the class constructor.

    Args:
        cls (type): Class whose constructor to inspect.
        kwargs (dict): Dictionary of keyword arguments.

    Returns:
        dict: Filtered dictionary containing only valid kwargs.
    """
    params = signature(cls.__init__).parameters
    return {k: v for k, v in kwargs.items() if k in params}



def make_model(name, **kwargs):
    """ Create a model instance from the registry.

    Args:
        name (str): Name of the model architecture to create.

    Returns:
        nn.Module: Instantiated model.
    """
    assert name in REGISTRY, f"Model '{name}' not found in registry. Available models: {list(REGISTRY.keys())}"

    valid_kwargs = _filter_kwargs(REGISTRY[name], kwargs)

    return REGISTRY[name](**valid_kwargs)
