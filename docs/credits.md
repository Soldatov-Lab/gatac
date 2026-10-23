# Credits & Attributions

## AI-assisted development

Most of GATAC code was written with AI assistance, used responsibly under strict validation against the expected behavior of the original CPU-based tools being translated. The bulk of the core feature porting/translations and GPU optimization work was carried out with Claude Opus 4.6 and GPT-5.4, while Claude Sonnet 4.6 was used for secondary development tasks.

All AI-assisted implementations were tightly constrained to match reference outputs from [SnapATAC2](https://github.com/scverse/SnapATAC2) {cite:p}`Zhang2024`, [ArchR](https://www.archrproject.com/) {cite:p}`Granja2021`, [chromVAR](https://greenleaflab.github.io/chromVAR/index.html) {cite:p}`Schep2017`, [MACS3](https://github.com/macs3-project/MACS) {cite:p}`Zhang2008`, [AMULET](https://github.com/UcarLab/AMULET) {cite:p}`Thibodeau2021`, and [TOBIAS](https://github.com/loosolab/TOBIAS) {cite:p}`Bentsen2020`. Results were also thoroughly tested for reproducibility on the [Reproducibility](reproducibility) page and in the associated test suite.

## Algorithms

### AMULET

GATAC's doublet/multiplet detection (`gatac.pp.detect_doublets`) is a **direct
port** of the [**AMULET**](https://github.com/UcarLab/AMULET) algorithm of
{cite:t}`Thibodeau2021`.

The overlap-detection sweep-line, the per-cell Poisson scoring, the row-sum
Poisson repeat-inference, and the BH-FDR correction are translated
line-for-line from the upstream Python source (`FragmentFileOverlapCounter.py`
and `AMULET.py`). The data flow has been rewritten to stream GATAC's
parquet fragment files onto the GPU, where the overlaps of all cells of a
chromosome are found at once; the optional repeat-filter pass is applied at
the raw-read level rather than the overlap level. The original AMULET v1.1 release is run as an
external subprocess during the
[reproducibility](reproducibility) tests, where it agrees with GATAC exactly
(Jaccard 1.000, q-value Pearson r 1.000 on 13,735 cells × 22 autosomes).

The original AMULET tool is distributed under the
[GNU GPL v3.0](https://github.com/UcarLab/AMULET/blob/main/LICENSE). No AMULET
source code is bundled with or linked into GATAC.

### TOBIAS

GATAC's footprinting (`gatac.tobias.footprint`) and differential binding
(`gatac.tobias.bindetect`) are a **port** of
[**TOBIAS**](https://github.com/loosolab/TOBIAS) 0.17.5 (`ATACorrect`,
`ScoreBigwig`, `BINDetect`) of {cite:t}`Bentsen2020`. The algorithms are translated line for line and executed on the GPU;
TOBIAS is run as an external tool in the reproducibility tests. TOBIAS is
distributed under the
[MIT License](https://github.com/loosolab/TOBIAS/blob/master/LICENSE).

The motif match thresholds port `threshold_from_p` of
[MOODS](https://github.com/jhkorhonen/MOODS) {cite:p}`Korhonen2017` (GPL v3
or Biopython License, used here under the Biopython License), and site scanning follows MOODS's summation order.

### MACS3

GATAC's peak calling (`gatac.peak.call`) is a **port** of
[**MACS3**](https://github.com/macs3-project/MACS) `callpeak` {cite:p}`Zhang2008`
(narrow peaks without control, with sub-peak summits), as SnapATAC2
{cite:p}`Zhang2024` runs it. The pileups, the p-score series and its rounding,
the q-value table, the peak closing and the Savitzky-Golay summit caller are
translated from MACS3 3.0 and 3.1 and run on run-length pileups on the GPU;
MACS3 and SnapATAC2 are run as external tools in the reproducibility tests,
where every summit and score matches. MACS3 is distributed under the
BSD 3-Clause License (Copyright (c) 2025, macs3 project team); the full notice
is kept in `gatac/peak/_macs.py`.

## Third-party code

### MINPACK (via SciPy)

`gatac/tobias/_minpack_relu.cu` is a CUDA adaptation of SciPy's C translation of
MINPACK's `lmdif` (used by `scipy.optimize.curve_fit`), specialised to the
two-parameter ReLU fits of TOBIAS ATACorrect. SciPy is distributed under the
BSD 3-Clause License. This product includes software developed by the
University of Chicago, as Operator of Argonne National Laboratory (MINPACK,
Copyright 1999 University of Chicago). The full notices are kept in the source
file.

### scPrinter

GATAC's `chromvar` implementation originated from the
[scPrinter](https://github.com/buenrostrolab/scPrinter/) repository
{cite:p}`Hu2025` and
underwent additional work for GPU memory optimization and closer agreement
with the original chromVAR R results.

### gmacs

GATAC's GPU peak calling started from
[**gmacs**](https://github.com/latchbio-workflows/gmacs), a GPU-accelerated
implementation of the [MACS3](https://github.com/macs3-project/MACS) algorithm
{cite:p}`Zhang2008` developed by Harihara Subrahmaniam Muralidharan at
LatchBio, adapted to the GATAC Parquet/cuDF data model. It has since been
replaced by the MACS3 port above. gmacs is dual-licensed under
[CC0 1.0 Universal](https://github.com/latchbio-workflows/gmacs/blob/main/COPYING)
and the
[MIT License](https://github.com/latchbio-workflows/gmacs/blob/main/LICENSE.MIT).
