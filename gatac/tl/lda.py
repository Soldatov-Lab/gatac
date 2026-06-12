"""GPU-accelerated Mini-batch LDA for binarized peak matrices.

Implements Online Variational Bayes [1]_ with CuPy for GPU acceleration.
Uses a sparse-native E-step that operates directly on CSR non-zero
structure, avoiding materialization of dense ``(K × V)`` phi matrices
per cell — critical for large peak sets (e.g. 700 k ATAC peaks).

Designed for single-cell ATAC-seq data where peaks are binary.

References
----------
.. [1] Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013).
       Stochastic variational inference. *JMLR*, 14, 1303–1347.
"""
from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as cp_sparse
from cupyx.scipy.special import digamma as cp_digamma
import numpy as np
import scipy.sparse as sp
from tqdm.auto import tqdm


class MiniBatchLDA:
    """GPU-accelerated Mini-batch LDA via Online Variational Bayes.

    Learns latent topics from a binary cell-by-peak matrix using stochastic
    variational inference with CuPy GPU acceleration.  The E-step works
    directly on the CSR sparsity pattern so the per-cell working set is
    ``K × nnz_i`` instead of ``K × V``.

    Parameters
    ----------
    n_topics : int
        Number of topics.
    alpha : float or None
        Symmetric Dirichlet prior on cell-topic distributions.
        If ``None``, defaults to ``1 / n_topics``.
    eta : float or None
        Symmetric Dirichlet prior on topic-peak distributions.
        If ``None``, defaults to ``1 / n_topics``.
    batch_size : int
        Cells per mini-batch.
    n_epochs : int
        Full passes over the data.
    kappa : float
        Learning rate decay exponent in (0.5, 1].
    tau : float
        Learning rate offset (down-weights early updates).
    e_step_iters : int
        Coordinate-ascent iterations in each E-step.
    use_full_gpu_matrix : bool
        If ``True``, try to cache the full CSR input matrix on GPU.
        Disabled by default because very large matrices can exhaust GPU
        memory before training starts.
    seed : int
        Random seed.
    verbose : bool
        Print epoch progress.

    Examples
    --------
    >>> import gatac as ga
    >>> model = ga.tl.MiniBatchLDA(n_topics=20, n_epochs=10, verbose=True)
    >>> model.fit_transform(peak_adata.X, binarize=True)
    >>> # Inspect the most-weighted peaks per topic
    >>> top = model.top_peaks(peak_adata.var_names, n_top=20)
    """


    def __init__(
        self,
        n_topics=20,
        *,
        alpha=None,
        eta=None,
        batch_size=256,
        n_epochs=5,
        kappa=0.7,
        tau=64.0,
        e_step_iters=20,
        use_full_gpu_matrix=False,
        seed=0,
        verbose=True,
    ):
        self.n_topics = n_topics
        self._alpha = alpha
        self._eta = eta
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.kappa = kappa
        self.tau = tau
        self.e_step_iters = e_step_iters
        self.use_full_gpu_matrix = use_full_gpu_matrix
        self.seed = seed
        self.verbose = verbose

        self.lambda_ = None  # (K, V) variational topic parameters (numpy)
        self.components_ = None  # (K, V) normalised topic distributions
        self._lambda_gpu = None  # (K, V) on GPU during training

    # ------------------------------------------------------------------
    @property
    def alpha(self):
        return self._alpha if self._alpha is not None else 1.0 / self.n_topics

    @property
    def eta(self):
        return self._eta if self._eta is not None else 1.0 / self.n_topics

    # ------------------------------------------------------------------
    @staticmethod
    def _to_csr(X, binarize):
        """Ensure *X* is scipy CSR float32, optionally binarize."""
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        elif not sp.isspmatrix_csr(X):
            X = X.tocsr()
        if binarize:
            X = X.astype(np.float32, copy=True)
            X.eliminate_zeros()
            X.data[:] = 1.0
        else:
            X = X.astype(np.float32, copy=False)
        return X

    @staticmethod
    def _try_gpu_full(X_csr):
        """Move entire CSR to GPU; return ``None`` on OOM."""
        try:
            X_gpu = cp_sparse.csr_matrix(X_csr)
            cp.cuda.Stream.null.synchronize()
            return X_gpu
        except (
            cp.cuda.memory.OutOfMemoryError,
            cp.cuda.runtime.CUDARuntimeError,
            MemoryError,
        ):
            cp.get_default_memory_pool().free_all_blocks()
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _segment_sum_csr_rows(values, indptr):
        """Sum contiguous CSR row segments for all topic columns at once."""
        n_rows = indptr.size - 1
        out = cp.zeros((n_rows, values.shape[1]), dtype=values.dtype)
        nonempty_rows = cp.nonzero(indptr[1:] > indptr[:-1])[0]
        if nonempty_rows.size == 0:
            return out

        cumsum = cp.cumsum(values, axis=0)
        starts = indptr[:-1][nonempty_rows]
        ends = indptr[1:][nonempty_rows] - 1
        seg_sums = cumsum[ends]
        prev = cumsum[cp.maximum(starts - 1, 0)]
        seg_sums -= prev * (starts > 0)[:, None]
        out[nonempty_rows] = seg_sums
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _e_step(X_gpu, Elog_beta, alpha, n_iters, need_sstats=True):
        """Sparse-native E-step on GPU.

        Operates directly on CSR ``indptr`` / ``indices`` so the working
        set per cell is ``K × nnz_i`` instead of ``K × V``.

        Parameters
        ----------
        X_gpu : cupyx.scipy.sparse.csr_matrix (n, V)
        Elog_beta : cupy.ndarray (K, V)
        alpha : float
        n_iters : int
        need_sstats : bool
            Skip sufficient-statistics computation on the transform path.

        Returns
        -------
        gamma : cupy.ndarray (n, K)
        sstats : cupy.ndarray (K, V) or None
        """
        n, V = X_gpu.shape
        K = Elog_beta.shape[0]

        indptr = X_gpu.indptr
        indices = X_gpu.indices
        data = X_gpu.data

        # Gather topic logits for non-zero peaks as contiguous (nnz_total, K).
        Elog_beta_nz = Elog_beta.T[indices]

        # Map each stored value to its row id.
        nnz_total = int(indptr[-1])
        positions = cp.arange(nnz_total, dtype=cp.int32)
        cell_ids = (
            cp.searchsorted(indptr, positions, side='right').astype(cp.int32)
            - 1
        )

        # Initialise γ
        nnz_per_row = cp.diff(indptr).astype(cp.float32)
        gamma = cp.empty((n, K), dtype=cp.float32)
        gamma[:] = alpha
        gamma += nnz_per_row[:, None] / K

        for _ in range(n_iters):
            psi_g = cp_digamma(gamma)
            psi_g -= cp_digamma(gamma.sum(axis=1, keepdims=True))

            # (nnz_total, K)  log-unnormalised phi at non-zero positions
            log_phi = psi_g[cell_ids] + Elog_beta_nz
            log_phi -= log_phi.max(axis=1, keepdims=True)
            phi = cp.exp(log_phi)
            phi /= phi.sum(axis=1, keepdims=True)
            phi *= data[:, None]

            gamma[:] = alpha
            gamma += MiniBatchLDA._segment_sum_csr_rows(phi, indptr)

        if not need_sstats:
            return gamma, None

        # Recompute φ from converged γ for sufficient statistics
        psi_g = cp_digamma(gamma)
        psi_g -= cp_digamma(gamma.sum(axis=1, keepdims=True))
        log_phi = psi_g[cell_ids] + Elog_beta_nz
        log_phi -= log_phi.max(axis=1, keepdims=True)
        phi = cp.exp(log_phi)
        phi /= phi.sum(axis=1, keepdims=True)
        phi *= data[:, None]

        sstats = cp.zeros((K, V), dtype=cp.float32)
        col_ids = indices.astype(cp.int32)
        for k in range(K):
            sstats[k] = cp.bincount(
                col_ids, weights=phi[:, k], minlength=V
            )

        return gamma, sstats

    # ------------------------------------------------------------------
    def fit(self, X, binarize=True):
        """Fit the model.

        Parameters
        ----------
        X : array-like or sparse, shape (n_cells, n_peaks)
            Peak accessibility matrix.  If *binarize* is ``True`` (default)
            values are clipped to {0, 1}.
        binarize : bool
            Clip input values to binary before processing.

        Returns
        -------
        self
        """
        X_csr = self._to_csr(X, binarize)
        n_cells, n_peaks = X_csr.shape
        rng = np.random.default_rng(self.seed)

        # Optionally keep the full matrix on GPU; default is per-batch transfer.
        X_gpu_full = (
            self._try_gpu_full(X_csr) if self.use_full_gpu_matrix else None
        )
        if self.verbose:
            loc = "GPU" if X_gpu_full is not None else "CPU (per-batch transfer)"
            print(f"Data: {loc}  |  {n_cells:,} cells × {n_peaks:,} peaks")

        # Initialise λ on GPU with a Gamma draw + prior
        rng_cp = cp.random.default_rng(self.seed)
        self._lambda_gpu = (
            rng_cp.gamma(
                100.0, 1.0 / 100.0, size=(self.n_topics, n_peaks)
            ).astype(cp.float32)
            + self.eta
        )

        update_ct = 0
        alpha = float(self.alpha)

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n_cells)
            n_batches = int(np.ceil(n_cells / self.batch_size))
            perm_gpu = cp.asarray(perm) if X_gpu_full is not None else None

            pbar = tqdm(
                range(n_batches),
                desc=f"Epoch {epoch + 1}/{self.n_epochs}",
                unit="batch",
                leave=False,
                disable=not self.verbose,
            )
            for b in pbar:
                idx = perm[b * self.batch_size : (b + 1) * self.batch_size]
                actual_n = len(idx)

                if X_gpu_full is not None:
                    X_b = X_gpu_full[
                        perm_gpu[
                            b * self.batch_size : (b + 1) * self.batch_size
                        ]
                    ]
                else:
                    X_b = cp_sparse.csr_matrix(X_csr[idx])

                Elog_beta = cp_digamma(self._lambda_gpu) - cp_digamma(
                    self._lambda_gpu.sum(axis=1, keepdims=True)
                )

                _, sstats = self._e_step(
                    X_b, Elog_beta, alpha, self.e_step_iters
                )

                # Stochastic natural-gradient M-step
                rho = (self.tau + update_ct) ** (-self.kappa)
                lam_hat = self.eta + (n_cells / actual_n) * sstats
                self._lambda_gpu *= 1 - rho
                self._lambda_gpu += rho * lam_hat
                update_ct += 1

            pbar.close()

        self.lambda_ = cp.asnumpy(self._lambda_gpu)
        self.components_ = self.lambda_ / self.lambda_.sum(
            axis=1, keepdims=True
        )
        return self

    # ------------------------------------------------------------------
    def transform(self, X, binarize=True):
        """Project cells onto learned topics.

        Parameters
        ----------
        X : array-like or sparse, shape (n_cells, n_peaks)
        binarize : bool
            Clip input values to binary before processing.

        Returns
        -------
        theta : ndarray, shape (n_cells, n_topics)
            Normalised topic proportions per cell.
        """
        if self._lambda_gpu is None:
            self._lambda_gpu = cp.asarray(self.lambda_)

        X_csr = self._to_csr(X, binarize)
        n_cells = X_csr.shape[0]

        X_gpu_full = (
            self._try_gpu_full(X_csr) if self.use_full_gpu_matrix else None
        )

        Elog_beta = cp_digamma(self._lambda_gpu) - cp_digamma(
            self._lambda_gpu.sum(axis=1, keepdims=True)
        )

        alpha = float(self.alpha)
        gammas = []
        n_batches = int(np.ceil(n_cells / self.batch_size))
        pbar = tqdm(
            range(0, n_cells, self.batch_size),
            total=n_batches,
            desc="Transform",
            unit="batch",
            leave=False,
            disable=not self.verbose,
        )
        for start in pbar:
            end = min(start + self.batch_size, n_cells)
            if X_gpu_full is not None:
                X_b = X_gpu_full[start:end]
            else:
                X_b = cp_sparse.csr_matrix(X_csr[start:end])
            g, _ = self._e_step(
                X_b, Elog_beta, alpha, self.e_step_iters, need_sstats=False
            )
            gammas.append(cp.asnumpy(g))

        pbar.close()

        gamma = np.concatenate(gammas)
        return gamma / gamma.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    def fit_transform(self, X, binarize=True):
        """Fit and return topic proportions."""
        self.fit(X, binarize=binarize)
        return self.transform(X, binarize=binarize)

    # ------------------------------------------------------------------
    def top_peaks(self, var_names, n_top=20):
        """Return a DataFrame with the top-weighted peaks in each topic.

        Parameters
        ----------
        var_names : array-like
            Peak / variable names (e.g. ``adata.var_names``).
        n_top : int
            Number of top peaks per topic.

        Returns
        -------
        pd.DataFrame
            Columns ``Topic_0 … Topic_{K-1}``, rows are ranks.
        """
        import pandas as pd

        var_names = np.asarray(var_names)
        top_idx = np.argsort(self.components_, axis=1)[:, ::-1][:, :n_top]
        return pd.DataFrame(
            {
                f"Topic_{k}": var_names[top_idx[k]]
                for k in range(self.n_topics)
            }
        )


def lda(
    adata,
    n_topics=20,
    layer=None,
    binarize=True,
    batch_size=256,
    n_epochs=5,
    e_step_iters=20,
    key_saved="X_lda",
    seed=0,
    **kwargs,
):
    """
    Learn topics from a binarized peak matrix using GPU-accelerated mini-batch LDA.

    Uses Online Variational Bayes (Hoffman et al., 2013) with JAX.
    Optimised for single-cell ATAC-seq binary peak-by-cell matrices.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with a peak accessibility matrix.
    n_topics : int, optional (default: 20)
        Number of topics to learn.
    layer : str or None, optional (default: None)
        Key in ``adata.layers`` for the input matrix. If None, ``adata.X`` is used.
    binarize : bool, optional (default: True)
        Clip values to {0, 1} before processing.
    batch_size : int, optional (default: 256)
        Cells per mini-batch. Reduce for large peak sets to save GPU memory.
    n_epochs : int, optional (default: 5)
        Number of passes over the data.
    e_step_iters : int, optional (default: 20)
        Variational inference iterations per E-step.
    key_saved : str, optional (default: "X_lda")
        Key in ``adata.obsm`` where topic proportions are saved.
    seed : int, optional (default: 0)
        Random seed for reproducibility.
    **kwargs
        Additional keyword arguments passed to :class:`~fastools.tl.lda.MiniBatchLDA`.

    Returns
    -------
    MiniBatchLDA
        The fitted model. Results are also stored in-place:

        - ``adata.obsm[key_saved]``: topic proportions *(n_cells, n_topics)*
        - ``adata.varm['lda_topics']``: peak loadings per topic *(n_peaks, n_topics)*

    Examples
    --------
    >>> import gatac as ga
    >>> model = ga.tl.lda(adata, n_topics=20, n_epochs=10)
    >>> # Cell × topic proportions are stored in adata.obsm["X_lda"]
    >>> adata.obsm["X_lda"].shape
    (n_cells, 20)
    """

    X = adata.X if layer is None else adata.layers[layer]
    model = MiniBatchLDA(
        n_topics=n_topics,
        batch_size=batch_size,
        n_epochs=n_epochs,
        e_step_iters=e_step_iters,
        seed=seed,
        **kwargs,
    )
    adata.obsm[key_saved] = model.fit_transform(X, binarize=binarize)
    adata.varm["lda_topics"] = model.components_.T
    return model