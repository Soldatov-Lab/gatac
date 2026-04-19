# Acknowledgements

## Third-party code

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
