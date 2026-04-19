# Tutorials

```{toctree}
:hidden:
:maxdepth: 1

../../notebooks/01_fragment_preprocessing
../../notebooks/02_embedding_peak_calling
../../notebooks/03_motif_enrichment
../../notebooks/04_chromvar
../../notebooks/05_gsea
```

---

## Browse tutorials

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Fragment Processing
:shadow: none
:link: ../../notebooks/01_fragment_preprocessing
:link-type: doc

Convert raw fragments into a Parquet-backed workflow, compute QC summaries,
and inspect the features carried forward into downstream analysis.
:::

:::{grid-item-card} Spectral Embedding, Clustering & Peak Calling
:shadow: none
:link: ../../notebooks/02_embedding_peak_calling
:link-type: doc

Build a low-dimensional embedding, identify cell groups, call peaks by
cluster, and assemble a peak matrix for follow-up analyses.
:::

:::{grid-item-card} Motif Enrichment
:shadow: none
:link: ../../notebooks/03_motif_enrichment
:link-type: doc

Test cluster-specific peak sets for transcription factor motif enrichment and
inspect the strongest motif signals in accessible regions.
:::

:::{grid-item-card} chromVAR: TF Activity Scores
:shadow: none
:link: ../../notebooks/04_chromvar
:link-type: doc

Compute chromVAR deviation scores from motif annotations and visualize
transcription factor activity across cells and clusters.
:::

:::{grid-item-card} Preranked GSEA
:shadow: none
:link: ../../notebooks/05_gsea
:link-type: doc

Rank marker peaks, derive motif-linked gene sets, and run enrichment analysis
to summarize regulatory programs by cluster.
:::

::::

---

Analysis notebooks for GATAC are maintained in the companion
[**gatac-notebooks**](https://github.com/Soldatov-Lab/gatac-notebooks) repository
and are included here as a git submodule at `notebooks/` in the repo root.

Notebooks are rendered with [MyST-NB](https://myst-nb.readthedocs.io/) — you
can view them here or download and run them locally after installing GATAC.

---

## Setting up the notebooks

The notebooks are stored in a separate repository and linked as a git
submodule under `notebooks/` at the repo root.  To initialise:

```bash
git submodule update --init --recursive
```

To run notebooks locally:

```bash
# Install GATAC with full dependencies
uv sync --extra cuda12

# Run a notebook
uv run jupyter lab notebooks/01_fragment_preprocessing.ipynb
```

:::{note}
Pre-computed outputs are committed to the repository so the docs build does
not require a GPU.  Set `nb_execution_mode = "auto"` in `docs/conf.py` to
re-execute notebooks during the docs build.
:::
