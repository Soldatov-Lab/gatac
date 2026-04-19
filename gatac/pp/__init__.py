from .convert import make_parquet, make_parquet_batch
from .process import read_fragments_parquet, combine
from .tile import make_tile_matrix
from .gene import make_gene_matrix
from .features import select_features, select_features_multi
from .metrics import (
    load_tss_from_gtf, 
    compute_metrics,
    cleanup_gpu_memory
)
from .filter import filter_fragments
from .genome import HG38 as _HG38, HG19 as _HG19, MM10 as _MM10, MM39 as _MM39

#: GRCh38 / hg38 chromosome sizes.
HG38 = _HG38
#: GRCh37 / hg19 chromosome sizes.
HG19 = _HG19
#: GRCm38 / mm10 chromosome sizes.
MM10 = _MM10
#: GRCm39 / mm39 chromosome sizes.
MM39 = _MM39

__all__ = [
    "make_parquet",
    "make_parquet_batch",
    "make_tile_matrix",
    "make_gene_matrix",
    "read_fragments_parquet", 
    "select_features",
    "select_features_multi",
    "combine",
    "load_tss_from_gtf", 
    "compute_metrics",
    "cleanup_gpu_memory",
    "filter_fragments",
    "HG38",
    "HG19",
    "MM10",
    "MM39",
]
