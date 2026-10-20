"""
Dimensionality reduction: spectral embedding and latent semantic indexing of
the cell × feature matrix.
"""

from .spectral import spectral
from .lsi import (
    lsi,
    iterative_lsi,
    project_lsi,
    model_from_adata,
    LSIModel,
)

__all__ = [
    "spectral",
    "lsi",
    "iterative_lsi",
    "project_lsi",
    "model_from_adata",
    "LSIModel",
]
