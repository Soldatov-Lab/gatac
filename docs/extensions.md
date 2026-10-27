# Extensions

An extension is a separate Python package that builds on GATAC: a new
analysis, or a wrapper around a published method that takes GATAC's fragment
files and AnnData objects as input. It is installed on its own
(`pip install gatac-<name>`) and, once installed, is reached from GATAC
like a built-in namespace and command:

```python
import gatac as ga

ga.extensions()          # installed extensions
ga.chromafold.predict(adata, ...)
```

```bash
gatac chromafold --help
```

GATAC does not import extensions until they are used, so an extension may
depend on heavy packages (PyTorch, model weights) that core GATAC never needs.

---

## Declaring an extension

An extension declares up to two
[entry points](https://packaging.python.org/en/latest/specifications/entry-points/)
in its `pyproject.toml`:

```toml
[project]
name = "gatac-chromafold"
dependencies = ["gatac>=0.1"]

# gatac.chromafold imports gatac_chromafold
[project.entry-points."gatac.ext"]
chromafold = "gatac_chromafold"

# `gatac chromafold ...`
[project.entry-points."gatac.cli"]
chromafold = "gatac_chromafold.cli:register"
```

- **`gatac.ext`** names the module to expose as `gatac.<name>`. It is
  imported on first access. Core namespaces (`io`, `pp`, `dr`, `peak`,
  `motif`, `tobias`, `pl`, `gpu`) always take precedence over an extension of
  the same name.
- **`gatac.cli`** names a function that receives the `gatac` command's
  argparse subparsers and adds its own, with the handler in `func`, as the
  built-in commands do:

  ```python
  def register(subparsers):
      p = subparsers.add_parser("chromafold", help="Predict contact maps from scATAC-seq")
      p.add_argument("input", help="Fragments .parquet file")
      p.set_defaults(func=run)

  def run(args):
      ...
  ```

  An extension whose commands fail to load prints a warning and leaves the
  other commands working.

Name the distribution `gatac-<name>` and the module `gatac_<name>`.

---

## What to build on

Extensions should use only the public API below. Modules and functions whose
name starts with an underscore can change in any release.

| Need | Use |
|---|---|
| Fragments, streamed on the GPU by chromosome and cell | {class}`gatac.io.FragmentFile` (`iter_batches`, `barcodes`, `barcode_counts`, `register` for DuckDB) |
| Genome sequence | {class}`gatac.io.Fasta` (`fetch` for `chrom:start-end` names, `region_bytes`, `chrom`, `chrom_sizes`) |
| BED, GTF and cell metrics files | {func}`gatac.io.read_bed`, {func}`gatac.io.read_gtf`, {func}`gatac.io.read_metrics` |
| Chromosome sizes of a named genome | `gatac.pp.HG38`, `HG19`, `MM10`, `MM39`, or `Fasta.chrom_sizes` |
| Motifs | {func}`gatac.motif.read`, {class}`gatac.motif.DNAMotif` |
| Counting and host → device copies at genome scale | {func}`gatac.gpu.bincount`, {func}`gatac.gpu.to_device` |
| Cell × feature matrices, embeddings, peaks | the functions of `gatac.pp`, `gatac.dr` and `gatac.peak`, and the AnnData layout below |

A note on `gatac.gpu.bincount`: unweighted `cupy.bincount` crashes in CuPy 14
past ~5M bins, which genome-wide arrays exceed; GATAC's own test suite rejects
any call to it, and extensions should use the wrapper too.

### AnnData layout

GATAC functions read and write these keys; an extension that follows them
works on the output of any GATAC step, and its output works with the next.

| Where | Key | Content |
|---|---|---|
| `obs_names` | | Barcodes, with the file's barcode prefix (`gatac rename`), as {class}`~gatac.io.FragmentFile` returns them |
| `obs` | `source_file` | Fragment file of each cell, for matrices built from several files |
| `var_names` | | Genomic features as `chrom:start-end`, 0-based and half-open (BED); genes by name |
| `obsm` | `X_spectral`, `X_lsi`, `X_iterative_lsi` | Cell embeddings of {func}`gatac.dr.spectral`, {func}`gatac.dr.lsi`, {func}`gatac.dr.iterative_lsi` |
| `obsm` | `chromvar` | chromVAR deviations |
| `varm` | `motif_match`, `bg_peaks` | Motif matches and background peaks of {func}`gatac.motif.scan`, {func}`gatac.motif.sample_bg_peaks` |
| `uns` | `tile_stats`, `lsi`, `iterative_lsi`, `peaks`, ... | Parameters and summaries of the step of the same name |

Write an extension's own results under keys that start with its name
(`obsm["chromafold_..."]`, `uns["chromafold"]`), so they cannot collide with
GATAC's or another extension's.

---

## Testing an extension

Test against the GATAC version range you declare. GATAC's own
`tests/conftest.py` shows how to build a small FASTA, MEME file and fragment
Parquet file in a temporary directory, so that tests run in seconds without
downloaded data.
