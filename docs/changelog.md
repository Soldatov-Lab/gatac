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

---

## 0.1.0 — *initial release*

First public version of GATAC.
