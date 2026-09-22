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
