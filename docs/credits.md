# Credits & Attributions

## AI-assisted development

Most of GATAC code was written with AI assistance, used responsibly under strict validation against the expected behavior of the original CPU-based tools being translated. The bulk of the core feature porting/translations and GPU optimization work was carried out with Claude Opus 4.6 and GPT-5.4, while Claude Sonnet 4.6 was used for secondary development tasks.

All AI-assisted implementations were tightly constrained to match reference outputs from [SnapATAC2](https://github.com/scverse/SnapATAC2), [ArchR](https://www.archrproject.com/), [chromVAR](https://greenleaflab.github.io/chromVAR/index.html), and [MACS3](https://github.com/macs3-project/MACS). Results were also thoroughly tested for reproducibility on the [Reproducibility](../reproducibility/README) page and in the associated test suite.

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
