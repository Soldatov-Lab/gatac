# Changelog

## Unreleased

### Added

- GPU-accelerated TSS enrichment scoring with streaming Parquet support.
- Tile matrix builder with SnapATAC2-compatible `"unique"` count strategy.
- Gene activity matrix builder using paired-insertion counting.
- GPU feature selection (ArchR-style quantile filtering).
- Multi-sample streaming feature selection via `select_features_multi`.
- Fragment filtering with Polars query engine (`filter_fragments`).
- CLI: `convert`, `metrics`, `filter`, `tile`, `gene`, `features`, `combine`.
- Spectral embedding (`tl.spectral`).
- Peak calling, peak merging, and peak matrix construction.
- Marker peak detection with GPU-accelerated binomial test and BH correction.
- Motif scanning from MEME format files.
- chromVAR deviation scoring (fully GPU-accelerated).
- GPU-accelerated preranked GSEA (`tl.gsea_motif_enrichment`).
- Mini-batch LDA via Online Variational Bayes (`tl.lda`, `tl.MiniBatchLDA`).
- Built-in chromosome sizes for hg38, hg19, mm10, mm39.
- `tl.gsea_motif_enrichment` now also reports `lead_edge_idx`, the ranked-list
  indices of the leading-edge features (previously only the count was
  available).
- Regression-based motif enrichment (`tl.motif_enrichment_regression`),
  following MEIRLOP. Models motif presence across every region as a logistic
  function of a continuous per-region score, adjusting for sequence composition
  through the dinucleotide frequencies. This removes the two structural
  weaknesses of the matched-pool test in `tl.motif_enrichment`: no background
  pool has to be sampled, so the result cannot depend on how it was drawn, and
  no significance cut is applied, so effect size is preserved rather than
  collapsing >99% of regions into a single class. Scanning is chunked over
  regions and the logistic fits are batched IRLS on GPU, so the full region set
  is usable — 673k regions x 1,165 motifs runs end to end on one H100.
  Degenerate and separated fits are flagged in a `converged` column and given a
  p-value of 1.0 rather than being reported as non-significant.

### Changed

- Preranked GSEA now scores the running enrichment score at hit positions only
  instead of running a length-N cumulative sum per permutation. The extrema of
  the running score are pinned to hit positions, so this is an exact rewrite,
  not an approximation — enrichment scores are unchanged to float32 precision.
  Cost drops from O(N) to O(set size) per (set, permutation): 7-60x faster
  depending on shape, with the gain growing as the ranked list and the number
  of sets grow. Permutation counts of 10^5 are now routine, lifting the floor
  on resolvable p-values.
- `gs_batch_size` for `tl.gsea_motif_enrichment` now defaults to 128 (was 4),
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
