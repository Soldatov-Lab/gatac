# Tutorials

Analysis notebooks for GATAC are maintained in the companion
[**gatac-notebooks**](https://github.com/Soldatov-Lab/gatac-notebooks) repository
and are included here as a git submodule at `notebooks/` in the repo root.

Notebooks are rendered with [MyST-NB](https://myst-nb.readthedocs.io/) — you
can view them here or download and run them locally after installing GATAC.

---

```{toctree}
:maxdepth: 1

../../notebooks/01_fragment_preprocessing
../../notebooks/02_embedding_peak_calling
../../notebooks/03_motif_enrichment
../../notebooks/04_chromvar
../../notebooks/05_gsea
```

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
