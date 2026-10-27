# Changelog

## Unreleased

### Added

- Extensions: packages that declare a `gatac.ext` entry point are reached as
  `gatac.<name>` (imported on first use, listed by `gatac.extensions()`), and
  a `gatac.cli` entry point adds their commands to `gatac`. The new
  Extensions page lists the public API to build on and the AnnData keys GATAC
  reads and writes.
- `io.Fasta`, the genome reader of every function that reads sequence
  (motif enrichment and scanning, chromVAR, TOBIAS), replaces the per-region
  pyfaidx loops: same sequences, about 1.7x faster on 119k peaks, and
  gzipped FASTA files now work for `tobias.footprint` / `bindetect_sites`
  too. A gzipped FASTA is decompressed once per process instead of on every
  call (`chromvar` decompressed and indexed hg19 twice, `enrichment_regression`
  once per batch, ~30 s each), and a missing `.fai` is built with NumPy
  (GRCh38 in 4 s instead of 18 s), so pyfaidx is no longer a dependency. `io.read_bed`, `io.read_gtf` and `io.read_metrics` are public, and the
  CuPy workarounds (`bincount`, `to_device`) are public as `gatac.gpu`.
- TOBIAS footprinting on the GPU in `gatac.tobias`: `footprint` (Tn5 bias
  correction and footprint scores per pseudobulk, from the fragments parquet),
  `bindetect` and `bindetect_sites` (differential TF binding), and the
  `pl.bindetect_volcano` / `pl.footprint_aggregate` plots. The per-site table
  of `bindetect` keeps the footprint score and bound call per group;
  `BINDetectResult.site_log2fc(g1, g2)` gives the per-site log2FC of a
  comparison.
- Parquet fragment format written by `make_parquet` / `gatac convert`:
  chromosome and barcode ids with the barcode table (original names, fragment
  and read counts) inside the file, ZSTD, delta-encoded starts and fragment
  lengths, barcode ids and counts as byte-stream split integers. All
  fragment readers go through the new `io.FragmentFile` and join cells and
  chromosomes by integer lookups. The sample's barcode prefix is stored once,
  in the footer: readers name cells with it, and the barcode table keeps the
  names of the original file.
  On a 484M-fragment library (RTX 5070): `compute_metrics` 51 s -> 9 s,
  `detect_doublets` 56 s -> 13 s, `peak.make_matrix` 14 s -> 6 s,
  `make_tile_matrix` 49 s -> 30 s (the rest is host-side matrix assembly). The
  whole-file read of `read_fragments_parquet` (gene matrices) needs about a
  quarter of the GPU memory. `make_parquet` parses the TSV once and encodes
  the next row group while the writer compresses the current one; on PBMC 5k
  (101M fragments, 6 cores) it takes 28 s and `filter_fragments` 13 s.
- `make_parquet_batch` accepts one barcode prefix per input file, and warns
  when the files it wrote share cell names, which keeps them from being
  combined into one matrix.
- `io.set_barcode_prefix` / `gatac rename` set the barcode prefix of a
  fragment file by rewriting its footer only, in place or into a copy, so
  that samples converted without prefixes can be told apart before they are
  combined, whatever the files' size.
- `make_parquet_batch(file_prefix=...)` (`gatac convert --file-prefix`)
  prefixes the output file names, one prefix for all files or one per file.
  In both prefixes, `{sample}` stands for each file's sample name, the part
  of the input paths that differs between the files (`io.sample_names`):
  `gatac convert "runs/*/outs/atac_fragments.tsv.gz" --file-prefix
  "{sample}_" --barcode-prefix "{sample}_"` names each Cell Ranger run's
  file and cells after its directory. Each file's output and prefixes can
  also be given one per file, the same way from the command line and from
  Python: as lists in input order (`gatac convert a.tsv.gz b.tsv.gz -o
  a.parquet b.parquet --barcode-prefix A_ B_`,
  `make_parquet_batch(output_paths=[...], barcode_prefix=[...])`), or as a
  sample sheet with `fragments`, `output` and `barcode_prefix` columns
  (`gatac convert --samples samples.csv`,
  `make_parquet_batch(samples="samples.csv")`). `gatac convert` reports
  conversion errors as messages instead of tracebacks. Inputs that would be written to the
  same output path (Cell Ranger names every sample's file
  `atac_fragments.tsv.gz`) now raise instead of overwriting each other.
  Files are started largest first, so that the largest do not run alone at
  the end.
- `make_tile_matrix` takes a list of fragment files (`gatac tile` several
  files, with `-n`) and builds one matrix of all their cells, of every tile
  or of the `n_features` most accessible, without writing and reading back
  per-sample matrices. It runs three steps, also public:
  `pp.scan_tiles` reads the fragment files once and returns the cells and
  tiles with per-tile statistics (`n_cells`, `count`, `count_sq`) and no
  matrix; `select_features` chooses tiles from these statistics, as it does
  from a matrix, or any other criterion can set `var["selected"]`; and
  `pp.build_tile_matrix` reads the files again to build the matrix of the
  chosen tiles only, in memory or straight into an `.h5ad`. Cells can be
  dropped between the scan and the build.
- `make_gene_matrix` and `make_gene_score_matrix` take a list of fragment
  files too (`gatac gene` / `gatac genescore` several files, with `-o`) and
  build one matrix of all their cells, named and tagged with
  `obs["source_file"]` as the tile matrix's are. Given an output path, each
  file's rows are written as they are built, so the combined matrix is never
  held in memory; `gatac combine` is no longer needed after them.
- A matrix of several fragment files (`make_tile_matrix`, `scan_tiles`,
  `make_gene_matrix`, `make_gene_score_matrix`) needs distinct cell names
  across the files: files that share names raise before any fragment is
  read, with the files concerned and how to give each sample a prefix.
- `io.to_tsv` / `io.FragmentFile.to_tsv` write a Parquet fragment file back to
  a `fragments.tsv.gz` in file order, with the original barcodes (without the
  file's prefix), BGZF-compressed (tabix-indexable);
  `gatac convert` does the same for `.parquet` inputs. An existing output,
  such as the original fragments file, is only replaced with
  `overwrite=True` / `--overwrite`.
- GPU-accelerated TSS enrichment scoring with streaming Parquet support.
- Tile matrix builder with SnapATAC2-compatible `"unique"` count strategy.
- Gene activity matrix builder using paired-insertion counting.
- GPU feature selection (ArchR-style quantile filtering).
- Fragment filtering on a query of the cell metrics (`filter_fragments`).
- CLI: `convert`, `metrics`, `filter`, `tile`, `gene`, `combine`.
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

- `filter_fragments` runs on the GPU: cells are matched by barcode id instead
  of by name, and the kept fragments are read, renumbered, encoded and
  ZSTD-compressed by cuDF into the same file layout. Several files are
  filtered at once (`workers`), largest first, under RMM's
  stream-ordered memory pool unless another memory resource is set. It
  decoded every fragment's barcode to a string and wrote the files one at a
  time on one CPU thread.
- Barcode prefixes are set on the fragment files only (`make_parquet`,
  `filter_fragments`, `io.set_barcode_prefix`), so that every step names a
  cell the same way. `make_tile_matrix`, `make_gene_matrix`,
  `make_gene_score_matrix` and their commands no longer take
  `barcode_prefix`: it renamed the matrix's cells but not the files', and
  the peak steps then found none of those cells' fragments.
  `filter_fragments(barcode_prefix=...)` sets the prefix of the output, one
  for all files or one per file, and matches cells to the metrics on the
  input's names; it added the prefix before matching.
- `compute_metrics` takes a list of fragment files: the TSS table is loaded
  once and several files are processed at once, under the same memory pool,
  and their rows are concatenated in the order given. It no longer runs a
  garbage collection after every batch.
- `make_tile_matrix` keeps each batch's tile counts on the GPU and builds the
  matrix there, a slab of cells at a time, copying it to the host once; it
  moved every batch to the host and back, then summed all of them at once.
  The tile table is built once per genome and tile size, and no garbage
  collection runs between batches. It runs under the stream-ordered memory
  pool.
- `dr.spectral` in chunks uses the count matrix as it is: the IDF weights,
  row norms and degrees scale the vectors of each product instead of the
  matrix, which is never rewritten. Chunks stay on the GPU while they fit in
  half of its free memory, the others are streamed from page-locked host
  memory, and the kernels read the rows through the matrix's own `indptr`;
  the per-chunk float copies, the host round trips of the normalization and
  the CPU degree computation are gone.
- `make_gene_matrix` finds the transcript domains of each insertion with one
  binary search: the domain starts and ends cut each chromosome into segments
  that every domain covers a contiguous run of. It checked the last 100
  domains starting before every insertion, which also missed a long domain
  once more than 100 others started inside it. PBMC 5k: 19.3 s -> 6.5 s
  (SnapATAC2 34 s), same GPU memory.
- `make_gene_matrix` takes each chromosome's fragments as one slice of the
  fragments grouped by chromosome; it masked the whole file once per
  chromosome, after sorting it by barcode. The annotation is parsed once per
  file and options, the counts are cast to their dtype and the matrix is
  sorted on the GPU, and it runs under the stream-ordered memory pool. The
  output is unchanged.
- `peak.make_matrix` keeps each batch's (cell, peak) counts on the GPU and
  builds each file's rows there, as `make_tile_matrix` does; the matrix is
  assembled once on the host, in its final dtype. It converted every file's
  counts to a CSR matrix on the host and added it to the running total,
  which grew slower with every file. All chromosomes of a batch are counted
  at once, and it runs under the stream-ordered memory pool. The output is
  unchanged.
- `make_gene_matrix` returns the matrix without writing it when
  `output_path` is None, as `make_tile_matrix` does; it used to write
  `<input>_gene_matrix.h5ad`. `gatac gene` still writes there by default.
- `make_gene_matrix` names and counts genes as SnapATAC2's
  `make_gene_matrix` does: by gene name, so genes sharing a name (5S_rRNA,
  Y_RNA, ...) are one column holding, per cell, the count of the most
  accessible of their transcripts, and columns come in order of first
  appearance in the annotation. They were one column per gene ID, named
  `<name>_<id>` when names repeated, sorted by ID. With
  `id_type="transcript"` the columns are named by transcript ID. On PBMC 5k
  the matrix now equals SnapATAC2's entry for entry (60,606 genes; was 61,806,
  1,277 of them in 77 repeated names).
- Fragment files store the fragment length (`end - start`) instead of `end`,
  and lengths, barcode ids and counts (now uint32) as byte-stream split
  integers: PBMC 5k is 369 MB instead of 417 MB (29.2 bits per fragment
  instead of 33.0), converted as fast and read back identical. Every reader
  still gives `end` and a uint16 `count`. DuckDB 2.0 is now required:
  earlier versions cannot read byte-stream split integers, which
  `FragmentFile.register` relies on.

- Per-barcode fragment counts (`io.FragmentFile.barcode_counts` with `chroms`
  or `mito_chroms`, behind `compute_metrics`, `make_tile_matrix`,
  `filter_fragments` and `detect_doublets`) and `pl.fragment_size_distribution`
  stream the fragments onto the GPU instead of scanning them with DuckDB, with
  the same results. On PBMC 5k (101M fragments, L40S): counts 0.7 s -> 0.35 s,
  size distribution 1.6 s -> 0.3 s, at about 1 GB of GPU memory whatever the
  library size. `pl.fragment_size_distribution` now also checks the fragment
  format, like every other reader.
- Every function that takes `metrics` (`make_tile_matrix`, `make_gene_matrix`,
  `make_gene_score_matrix`, `filter_fragments`, `pl.qc_metrics`) accepts the
  same inputs: a CSV path, or a pandas, cuDF or Polars DataFrame. A metrics
  path that does not exist raises `FileNotFoundError`; `make_tile_matrix`,
  `make_gene_matrix` and `make_gene_score_matrix` used to warn and keep every
  barcode, skipping `filter_query`.
- Polars is no longer a dependency, nor `cudf-polars` in the `cuda12` /
  `cuda13` extras (it was never used). GTF/GFF3 annotations are read by one
  DuckDB reader for `compute_metrics`, `make_gene_matrix` and
  `make_gene_score_matrix`: as fast as Polars on GENCODE v44 (1.7 s) with
  0.5 GB of host memory instead of 5.1 GB. `compute_metrics(tss_source=)`
  takes a pandas DataFrame (cuDF and Polars ones are converted).
- Tied rows now keep a defined order where Polars left an arbitrary one:
  annotation rows keep the file order, and `peak.markers` lists features with
  equal FDR and |log2FC| in `var` order. On PBMC 5k this moves at most 2 of
  the top 200 markers of a group, 0-5 of 1,165 motifs across the 0.05
  threshold of `motif.enrichment`, and changes 58 of 53M entries (4 genes with
  over 100 overlapping transcripts) of the `make_gene_matrix` notebook matrix;
  the reproducibility gene matrix still matches SnapATAC2 exactly.

- `peak.call` runs MACS3 `callpeak` on the GPU and gives the peaks of
  SnapATAC2's `macs3`: on PBMC 5k (3 groups, 235k summits) every summit's
  start, end, position, score, fold enrichment and p- and q-value is
  identical, in 5.7 s against 140 s for SnapATAC2 (24x). Pileups are kept as
  runs between breakpoints instead of one value per base, so the time follows
  the number of fragments instead of the genome length (~16 s per group
  before, whatever its depth), and the files are read once for all the
  groups that fit in host memory (new `host_memory_gb`, by default half of
  what the job or machine has left). Host memory beyond the fragments read
  stays at about 20 GB for a 1.4e9-fragment group. Results differ from the previous gmacs port: one row per summit
  (`call_summits=True`, as SnapATAC2), MACS3's peak boundaries and scores.
  New `shift`, `call_summits` and `macs3_version` (`"3.1"` for the summit
  filtering of MACS3 3.1) parameters; the other parameters keep their names
  and map to the MACS3 options in the docstring.

- Fragment Parquet files written before this format are no longer read: every
  fragment reader raises "not a GATAC fragment file" on them. Convert the
  original `fragments.tsv.gz` again with `make_parquet` / `gatac convert`.
- `make_parquet` and `gatac convert` no longer take `chroms` (a fixed
  chromosome id order): ids follow the order of first appearance in each
  file, and readers map them to names through the file's own chromosome list.
- `detect_doublets` drops duplicate entries in `barcodes`, with a warning;
  each barcode used to be scored once per occurrence.
- Fragment file I/O moved to a new `gatac.io` namespace: `make_parquet`,
  `make_parquet_batch`, `FragmentFile` and `read_fragments_parquet` are no
  longer in `gatac.pp`.
- `gatac.tl` is split by mode of analysis into `gatac.dr` (spectral, LSI),
  `gatac.peak` (peak calling, peak matrix, marker peaks) and
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
- `make_gene_matrix` processes all cells together, one chromosome at a time,
  instead of once per batch of cells and chromosome; the matrix is unchanged
  and `cell_batch_size` is ignored. PBMC 5k (14,225 cells x 61,806 genes):
  34.5 s -> 22.6 s.

### Fixed

- `peak.call` and `peak.make_matrix` given a directory found none of the
  files of a tile matrix built from several fragment files, whose
  `source_file` holds the Parquet file name: they looked for
  `<name>.parquet.parquet`, warned that each file was missing and returned no
  peaks, or an empty matrix.
- `dr.spectral` in chunks (any `chunk_size`, and by default on large
  datasets) replaced `adata.X` with the normalized float32 matrix when no
  feature subset was taken (`features=None`).
- `dr.spectral` streaming the matrix in chunks (any `chunk_size`, and by
  default on large datasets) read int64 indices as int32, which scipy uses
  once the number of non-zeros exceeds the int32 range, and gave most entries
  the wrong IDF weight. The embedding now matches SnapATAC2's.
- `make_tile_matrix`, `make_gene_matrix` and `make_gene_score_matrix` accept
  a pandas `metrics` DataFrame (raised `TypeError`).
- BED inputs (`peak.make_matrix(peak_file=)`, `detect_doublets(repeat_filter=)`)
  keep chromosome names as strings and skip `#`, `track` and `browser` lines,
  as `tobias.footprint` already did: chromosomes named `1`, `2`, ... were read
  as integers and matched no fragment.
- The `peak.markers` example filtered its pandas result with Polars syntax.
- `motif.compute_deviations` samples background peaks as chromVAR does:
  each background iteration replaces every motif peak by its background peak.
  It summed the counts of the peaks whose background peak matched the motif
  instead, which is a different set because background peaks repeat. The
  spread of the background deviations is now the sample standard deviation,
  as R's `sd()`. Given R's motif matches and background peaks, the z-scores
  equal R chromVAR's up to float32 rounding (correlation 1.000000, max
  difference 0.002, was 0.975); same speed.
- `motif.enrichment` matches the same regions as SnapATAC2's
  `motif_enrichment` again, so fold changes and p-values agree to rounding: it
  turns `pvalue` into a score threshold with SnapATAC2's score distribution
  and gives zero probabilities SnapATAC2's 1e-4 pseudocount. It had been
  using the right-tail threshold and 1e-10 floor written for chromVAR's
  motifmatchr parity, which keep fewer sites of motifs with few distinct
  scores; on PBMC 5k, 230 of 1,165 cisBP motifs matched other regions and
  the fold-change correlation fell to 0.978. `motif.scan` and
  `presence_matrix` are unchanged. The thresholds are computed on all
  allocated cores (1,165 motifs in about 2 s on 6).

- `peak.merge` gives SnapATAC2's `merge_peaks` peaks: among equally
  significant overlapping peaks it keeps the last in position order, as
  SnapATAC2 does, and clips to the chromosome after merging, with the summit
  offset of peaks clipped at position 0. 147 of 119,141 merged peaks differed
  on PBMC 5k.

- `make_gene_matrix` gives the same gene coordinates on every run: `var`
  `start`/`end` are the span of the gene's transcript domains on its
  chromosome, and `chrom`, `strand` and the names come from its first
  transcript in the annotation. They came from an arbitrary transcript, which
  changed between runs for ~2% of genes. The matrix is unchanged.
- `tobias.bindetect` needs far less host memory: genome-wide on 5 pseudobulks
  with the 3,935 cisBP motifs (102M binding sites) it ran out of memory on a
  23 GB machine. Per-site scores are normalised one group at a time in chunks,
  TOBIAS's site order is computed once for all comparisons, the per-site
  log2FCs are no longer stored, and the null is sampled in smaller chunks.
  On 34M sites and 5 groups the call adds 3.6 GB instead of 10.4 GB. Results
  are unchanged.
- `tobias.footprint` peaks at 9.3 GB of host memory instead of 19.9 GB
  genome-wide on 5 pseudobulks, with identical tracks. Reading the cut sites
  deduplicated fragments with a polars `unique` over all rows (13 GB of hash
  tables for a 1.2 GB table), and polars' allocator kept ~5 GB resident
  afterwards; fragments are now split by group and chromosome on the GPU and
  deduplicated per part by a sort, and the tracks are copied straight into
  their output arrays.
- Large host-to-device copies in `bindetect` and `bindetect_sites` go in
  64 MB chunks: with the WSL2 CUDA driver, each copy of about 1 GB or more left
  a host buffer of its size allocated for the rest of the process.
- `make_gene_matrix` with a `metrics` table that has no `n_unique` column
  (e.g. only barcodes) kept no cells and failed with "blocks must be 2-D";
  the listed cells are now kept, with their fragment counts read from the file.
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
