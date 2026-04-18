"""
Fragment processing pipeline for generating tile and gene matrices.
"""
from __future__ import annotations

import logging
import time
import gc
from pathlib import Path
from typing import Optional, Tuple, List, Literal

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from tqdm import tqdm

from .genome import HG38, HG19, MM10, MM39  # For type checking if needed in other modules, but actually not used here.
# Actually, let's just remove the .tile and .gene imports.

logger = logging.getLogger(__name__)

# Predefined dtypes to minimize memory footprint
FRAGMENT_DTYPES = {
    'chrom': 'category',
    'start': 'uint32',
    'end': 'uint32',
    'barcode': 'category',
    'count': 'uint16'
}

def read_fragments_parquet(
    input_path: str | Path,
    low_memory: bool = True,
    columns: Optional[List[str]] = None,
) -> cudf.DataFrame:
    """
    Read ATAC fragments from Parquet file optimized for GPU memory.

    Note: Parquet files contain intrinsic schema metadata. This function 
    expects the file to contain columns matching FRAGMENT_DTYPES keys:
    ['chrom', 'start', 'end', 'barcode', 'count'].
    """
    if columns is None:
        columns = list(FRAGMENT_DTYPES.keys())
        
    with cudf.option_context("io.parquet.low_memory", low_memory):
        df = cudf.read_parquet(str(input_path), columns=columns)
        
        # Ensure dtypes match our expectation to save memory
        for col, dtype in FRAGMENT_DTYPES.items():
            if col in df.columns and df[col].dtype != dtype:
                df[col] = df[col].astype(dtype)
                
    return df





def combine(
    input_paths: list[str | Path],
    output_path: str | Path,
):
    """
    Merge multiple h5ad files into a single file with efficient streaming.

    Parameters
    ----------
    input_paths : list of str or Path
        List of paths to h5ad files
    output_path : str or Path
        Output path for combined h5ad file
    """
    input_paths = [Path(p) for p in input_paths]
    output_path = Path(output_path)

    if len(input_paths) == 0:
        raise ValueError("No input files provided")

    if len(input_paths) == 1:
        logger.info("Single file provided, copying to output")
        adata = sc.read_h5ad(str(input_paths[0]))
        adata.write_h5ad(str(output_path))
        return

    logger.info(f"Combining {len(input_paths)} h5ad files")

    # Get reference var from first file
    first_adata = sc.read_h5ad(str(input_paths[0]), backed='r')
    n_vars = first_adata.n_vars
    var_df = first_adata.var.copy()
    
    # Determine the highest dtype across all inputs and collect cell counts
    logger.info("Checking feature consistency and determining optimal dtype...")
    total_cells = 0
    total_nnz = 0
    dtypes = []

    for fpath in tqdm(input_paths, desc="Scanning files"):
        adata = sc.read_h5ad(str(fpath), backed='r')
        if adata.n_vars != n_vars:
            raise ValueError(
                f"Feature mismatch: {fpath.name} has {adata.n_vars} features, "
                f"expected {n_vars}"
            )
        
        # Collect dtype
        if sp.issparse(adata.X):
            dtypes.append(adata.X.dtype)
            total_nnz += adata.X.nnz
        else:
            dtypes.append(adata.X.dtype)
            total_nnz += np.count_nonzero(adata.X)
            
        total_cells += adata.n_obs
        del adata

    # Select the "highest" dtype
    optimal_dtype = np.result_type(*dtypes)
    logger.info(f"Using combined dtype: {optimal_dtype}")

    # =========================================================================
    # Build combined sparse matrix
    # =========================================================================
    logger.info("Building combined matrix...")

    data_list = []
    indices_list = []
    indptr_list = []
    obs_list = []
    
    current_nnz = 0
    total_cells_processed = 0

    for fpath in tqdm(input_paths, desc="Stream loading"):
        adata = sc.read_h5ad(str(fpath))
        n_obs = adata.n_obs
        
        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        elif not isinstance(X, sp.csr_matrix):
            X = X.tocsr()

        nnz = X.nnz
        
        # Collect matrix components with optimal dtype
        data_list.append(X.data.astype(optimal_dtype))
            
        # Determine optimal index dtype
        index_dtype = np.uint32 if n_vars > 65535 else np.uint16
        indices_list.append(X.indices.astype(index_dtype))
        
        # Adjust indptr for concatenation
        if total_cells_processed == 0:
            indptr_list.append(X.indptr.astype(np.uint64))
        else:
            # Drop the first 0 to append to existing indptr
            chunk_indptr = X.indptr[1:].astype(np.uint64)
            indptr_list.append(chunk_indptr + current_nnz)

        # Collect obs metadata
        obs_df = adata.obs.copy()
        obs_df['source_file'] = fpath.name
        obs_list.append(obs_df)

        current_nnz += nnz
        total_cells_processed += n_obs
        
        del adata, X
        gc.collect()

    # =========================================================================
    # Final Assembly
    # =========================================================================
    logger.info("Final assembly...")
    
    # Concatenate sparse components
    all_data = np.concatenate(data_list)
    del data_list
    all_indices = np.concatenate(indices_list)
    del indices_list
    all_indptr = np.concatenate(indptr_list)
    del indptr_list
    gc.collect()

    # Build final sparse matrix
    combined_X = sp.csr_matrix(
        (all_data, all_indices, all_indptr),
        shape=(total_cells_processed, n_vars)
    )
    del all_data, all_indices, all_indptr
    gc.collect()

    # Build combined obs
    combined_obs = pd.concat(obs_list)
    del obs_list
    
    # Make barcodes unique
    combined_obs.index.name = 'barcode'
    if 'barcode' in combined_obs.columns:
        combined_obs.drop(columns=['barcode'], inplace=True)
    combined_obs.reset_index(inplace=True)

    if not combined_obs['barcode'].is_unique:
        n_dups = combined_obs['barcode'].duplicated().sum()
        dup_examples = combined_obs['barcode'][combined_obs['barcode'].duplicated(keep=False)].unique()[:5].tolist()
        raise ValueError(
            f"Detected {n_dups:,} duplicate cell barcode(s) across the input files "
            f"(e.g. {dup_examples}). "
            "Barcodes must be unique before combining. "
            "Re-run the preceding processing steps with a per-sample `--barcode-prefix` "
            "argument (e.g. `gatac convert sample.tsv.gz --barcode-prefix sample1`) "
            "so that each sample's barcodes are namespaced and collisions are avoided."
        )
    else:
        combined_obs.index = combined_obs['barcode'].values
        combined_obs.drop(columns=['barcode'], inplace=True)

    # Build combined AnnData
    combined_adata = ad.AnnData(
        X=combined_X,
        obs=combined_obs,
        var=var_df,
    )

    # Save
    combined_adata.write_h5ad(str(output_path))
    logger.info(f"Saved combined matrix ({total_cells_processed :,} cells × {n_vars:,} features) to {output_path}")
