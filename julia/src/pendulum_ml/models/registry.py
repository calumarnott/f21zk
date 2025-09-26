""" Registry for models. This is where you can register new model architectures.

Example usage:
```
model = make_model("mlp", in_dim=2, hidden=(128,128), out_dim=1, dropout=0.0)
```
"""

from .mlp import MLP
# add other models here as needed

REGISTRY = {
  "mlp": MLP,
  # add other models here as needed
}

def make_model(name, **kwargs):
    """ Create a model instance from the registry.

    Args:
        name (str): Name of the model architecture to create.

    Returns:
        nn.Module: Instantiated model.
    """
    return REGISTRY[name](**kwargs)
