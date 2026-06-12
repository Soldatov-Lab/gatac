# Credits & Attributions

## AI-assisted development

Most of GATAC code was written with AI assistance, used responsibly under strict validation against the expected behavior of the original CPU-based tools being translated. The bulk of the core feature porting/translations and GPU optimization work was carried out with Claude Opus 4.6 and GPT-5.4, while Claude Sonnet 4.6 was used for secondary development tasks.

All AI-assisted implementations were tightly constrained to match reference outputs from [SnapATAC2](https://github.com/scverse/SnapATAC2), [ArchR](https://www.archrproject.com/), [chromVAR](https://greenleaflab.github.io/chromVAR/index.html), [MACS3](https://github.com/macs3-project/MACS), and [AMULET](https://github.com/UcarLab/AMULET). Results were also thoroughly tested for reproducibility on the [Reproducibility](reproducibility) page and in the associated test suite.

## Algorithms

### AMULET

GATAC's doublet/multiplet detection (`gatac.pp.detect_doublets`) is a **direct
port** of the [**AMULET**](https://github.com/UcarLab/AMULET) algorithm of
Thibodeau A. et al., *Genome Biology*, 2021.

The overlap-detection sweep-line, the per-cell Poisson scoring, the row-sum
Poisson repeat-inference, and the BH-FDR correction are translated
line-for-line from the upstream Python source (`FragmentFileOverlapCounter.py`
and `AMULET.py`). The data flow has been rewritten to operate on GATAC's
parquet fragment files via DuckDB and to parallelize per-chromosome with a
worker pool; the optional repeat-filter pass is applied at the raw-read level
rather than the overlap level. The original AMULET v1.1 release is run as an
external subprocess during the
[reproducibility](reproducibility) tests, where it agrees with GATAC exactly
(Jaccard 1.000, q-value Pearson r 1.000 on 13,735 cells × 22 autosomes).

The original AMULET tool is distributed under the
[GNU GPL v3.0](https://github.com/UcarLab/AMULET/blob/main/LICENSE). No AMULET
source code is bundled with or linked into GATAC.

## Third-party code

### scPrinter

GATAC's `chromvar` implementation originated from the
[scPrinter](https://github.com/buenrostrolab/scPrinter/) repository and
underwent additional work for GPU memory optimization and closer agreement
with the original chromVAR R results.

### gmacs

Portions of GATAC's GPU-accelerated peak-calling code are derived from
[**gmacs**](https://github.com/latchbio-workflows/gmacs) — a GPU-accelerated
implementation of the [MACS3](https://github.com/macs3-project/MACS) algorithm developed by Harihara Subrahmaniam Muralidharan at
LatchBio.

The relevant code has been extensively modified for integration into the GATAC pipeline. Key updates include adaptation to the GATAC Parquet/cuDF data model, enhanced memory management through streaming, and a refactored API designed to align with the scverse ecosystem.

gmacs is dual-licensed under
[CC0 1.0 Universal](https://github.com/latchbio-workflows/gmacs/blob/main/COPYING)
and the
[MIT License](https://github.com/latchbio-workflows/gmacs/blob/main/LICENSE.MIT).
