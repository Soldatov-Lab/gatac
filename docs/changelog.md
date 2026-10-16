# Changelog

## Unreleased

### Added

- TOBIAS footprinting on the GPU in `gatac.tobias`: `footprint` (Tn5 bias
  correction and footprint scores per pseudobulk, from the fragments parquet),
  `bindetect` and `bindetect_sites` (differential TF binding), and the
  `pl.bindetect_volcano` / `pl.footprint_aggregate` plots.
- Parquet fragment format written by `make_parquet` / `gatac convert`:
  chromosome and barcode ids with the barcode table (original names, fragment
  and read counts) inside the file, ZSTD and delta-encoded coordinates. All
  fragment readers go through the new `io.FragmentFile` and join cells and
  chromosomes by integer lookups.
  On a 484M-fragment library (RTX 5070): `compute_metrics` 51 s -> 9 s,
  `detect_doublets` 56 s -> 13 s, `peak.make_matrix` 14 s -> 6 s,
  `make_tile_matrix` 49 s -> 30 s (the rest is host-side matrix assembly). The
  whole-file read of `read_fragments_parquet` (gene matrices) needs about a
  quarter of the GPU memory.
- `make_parquet_batch` accepts one barcode prefix per input file.
- `io.to_tsv` / `io.FragmentFile.to_tsv` write a Parquet fragment file back to
  a `fragments.tsv.gz` in file order, BGZF-compressed (tabix-indexable);
  `gatac convert` does the same for `.parquet` inputs.
- GPU-accelerated TSS enrichment scoring with streaming Parquet support.
- Tile matrix builder with SnapATAC2-compatible `"unique"` count strategy.
- Gene activity matrix builder using paired-insertion counting.
- GPU feature selection (ArchR-style quantile filtering).
- Multi-sample streaming feature selection via `select_features_multi`.
- Fragment filtering with Polars query engine (`filter_fragments`).
- CLI: `convert`, `metrics`, `filter`, `tile`, `gene`, `features`, `combine`.
- Spectral embedding (`dr.spectral`).
- Latent semantic indexing — GPU TF-IDF plus truncated SVD (`dr.lsi`),
  reproducing ArchR's `.computeLSI` for all three `LSIMethod` variants,
  with projection of new cells onto a fitted embedding (`dr.project_lsi`).
- Iterative LSI (`dr.iterative_lsi`), a port of ArchR's `addIterativeLSI`:
  iterative feature selection driven by per-cluster accessibility variance,
  with GPU SNN + Leiden/Louvain clustering between rounds. Adds a cuGraph
  dependency to the CUDA extras; `dr.lsi` itself does not require it.
- Peak calling, peak merging, and peak matrix construction.
- Marker peak detection with GPU-accelerated binomial test and BH correction.
- Motif scanning from MEME format files.
- chromVAR deviation scoring (fully GPU-accelerated).
- GPU-accelerated preranked GSEA (`motif.gsea`).
- Mini-batch LDA via Online Variational Bayes (`dr.lda`, `dr.MiniBatchLDA`).
- Built-in chromosome sizes for hg38, hg19, mm10, mm39.
- `motif.gsea` now also reports `lead_edge_idx`, the ranked-list
  indices of the leading-edge features (previously only the count was
  available).
- Regression-based motif enrichment (`motif.enrichment_regression`),
  following MEIRLOP. Models motif presence across every region as a logistic
  function of a continuous per-region score, adjusting for sequence composition
  through the dinucleotide frequencies. This removes the two structural
  weaknesses of the matched-pool test in `motif.enrichment`: no background
  pool has to be sampled, so the result cannot depend on how it was drawn, and
  no significance cut is applied, so effect size is preserved rather than
  collapsing >99% of regions into a single class. Scanning is chunked over
  regions and the logistic fits are batched IRLS on GPU, so the full region set
  is usable — 673k regions x 1,165 motifs runs end to end on one H100.
  Degenerate and separated fits are flagged in a `converged` column and given a
  p-value of 1.0 rather than being reported as non-significant.

### Changed

- Fragment file I/O moved to a new `gatac.io` namespace: `make_parquet`,
  `make_parquet_batch`, `FragmentFile` and `read_fragments_parquet` are no
  longer in `gatac.pp`.
- `gatac.tl` is split by mode of analysis into `gatac.dr` (spectral, LSI,
  LDA), `gatac.peak` (peak calling, peak matrix, marker peaks) and
  `gatac.motif` (motif enrichment, GSEA, chromVAR). Motif functions drop the
  `motif` from their names: `read`, `scan`, `enrichment`, `presence_matrix`,
  `enrichment_regression` and `gsea` (were `read_motifs`, `scan_motifs`,
  `motif_enrichment`, `motif_presence_matrix`, `motif_enrichment_regression`
  and `gsea_motif_enrichment`); peak functions likewise drop `peak`: `call`,
  `merge`, `make_matrix`, `markers` and `get_markers` (were `call_peaks`,
  `merge_peaks`, `make_peak_matrix`, `marker_peaks` and `get_marker_peaks`;
  results still go to `adata.uns["marker_peaks"]`). Only user-facing functions
  are listed: the iterative-LSI stages (`feature_accessibility`,
  `initial_features`, `accessibility_pool`, `cluster_var_features`,
  `scale_dims`) are imported from `gatac.dr.lsi`, and `parse_meme` from
  `gatac.motif.motif`.
- Preranked GSEA now scores the running enrichment score at hit positions only
  instead of running a length-N cumulative sum per permutation. The extrema of
  the running score are pinned to hit positions, so this is an exact rewrite,
  not an approximation — enrichment scores are unchanged to float32 precision.
  Cost drops from O(N) to O(set size) per (set, permutation): 7-60x faster
  depending on shape, with the gain growing as the ranked list and the number
  of sets grow. Permutation counts of 10^5 are now routine, lifting the floor
  on resolvable p-values.
- `gs_batch_size` for `motif.gsea` now defaults to 128 (was 4),
  since the scoring block no longer scales with the ranked-list length.

### Fixed

- Preranked GSEA no longer double-counts features that appear more than once
  within the same feature set; duplicates previously inflated both the size
  filter and the enrichment-score normalisation.
- Preranked GSEA returned a wrong enrichment score when a feature set's members
  all had a ranking metric of exactly zero, missing the true extremum at the end
  of the ranked list.
- FDR computation no longer upcasts the whole null distribution once per feature
  set, which dominated runtime for large feature-set collections (37s of a 38s
  run at 5000 sets). Results are bit-identical.

---

## 0.1.0 — *initial release*

First public version of GATAC.
