# Installation

## Prerequisites

GATAC requires **Python ≥ 3.11** and a CUDA-capable GPU with drivers
compatible with CUDA 12 or 13.

The following system libraries must be available:

- CUDA Toolkit 12.x or 13.x
- cuDNN (optional, for future neural components)

---

## Quick install with uv

```bash
# Clone the repository
git clone https://github.com/Soldatov-Lab/gatac.git
cd GATAC

# Install with CUDA 12 support
uv sync --extra cuda12

# Or with CUDA 13 support
uv sync --extra cuda13
```

After installation the `gatac` command will be available:

```bash
gatac --help
```

---

## Install from PyPI *(planned)*

A PyPI release is planned for a future version.  For now, install directly
from the repository as shown above.

---

## Optional: documentation dependencies

To build the docs locally, install the docs extras:

```bash
uv sync --extra docs
```

Then build with:

```bash
cd docs
make html
```

---

## Verifying your installation

```python
import gatac as ga
print(ga.__version__)

# Check that GPU is available
import cudf
print(cudf.get_device_info())
```

---

## Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 8 GB    | 24–80 GB    |
| RAM       | 32 GB   | 128+ GB     |
| Storage   | —       | NVMe SSD    |

:::{note}
GATAC can run on CPU-only systems by omitting the `cuda12`/`cuda13` extras
and relying on Polars CPU execution, but performance will be significantly
reduced and some GPU-only features will be unavailable.
:::

---

## Conda / Mamba

GATAC is designed around `uv` for reproducible installs.  Conda support may
be added in a future release.  If you need Conda, you can install the RAPIDS
dependencies manually:

```bash
conda create -n gatac -c rapidsai -c conda-forge -c nvidia \
    rapids=24.02 python=3.11 cuda-version=12.0
conda activate gatac
pip install -e .
```
