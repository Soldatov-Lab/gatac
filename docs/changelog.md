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

---

## 0.1.0 — *initial release*

First public version of GATAC.
