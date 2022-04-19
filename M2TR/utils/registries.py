from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""

LOSS_REGISTRY = Registry("LOSS")
LOSS_REGISTRY.__doc__ = """
Registry for loss functions.
The registered object will be called with `obj(cfg)`.
"""


DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets.
The registered object will be called with `obj(cfg)`.
"""
