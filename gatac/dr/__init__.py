"""
Dimensionality reduction: spectral embedding, latent semantic indexing and
topic modelling of the cell × feature matrix.
"""

from .spectral import spectral
from .lsi import (
    lsi,
    iterative_lsi,
    project_lsi,
    model_from_adata,
    LSIModel,
)
from .lda import lda, MiniBatchLDA

__all__ = [
    "spectral",
    "lsi",
    "iterative_lsi",
    "project_lsi",
    "model_from_adata",
    "LSIModel",
    "lda",
    "MiniBatchLDA",
]
