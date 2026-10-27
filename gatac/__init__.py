"""
GATAC - GPU-accelerated ATAC-seq processing toolkit.

Public API:
    - io: Fragment files (reading, writing, conversion)
    - pp: Preprocessing functions
    - dr: Dimensionality reduction
    - peak: Peak calling and marker peaks
    - motif: Motif enrichment and chromVAR
    - tobias: TOBIAS footprinting and differential TF binding
    - pl: Plotting
    - gpu: GPU helpers for code built on GATAC

Installed extensions (packages with a ``gatac.ext`` entry point) are reached
as ``gatac.<name>``, imported on first use; see :func:`extensions`.
"""

import logging
import os
import sys
from importlib.metadata import EntryPoint, entry_points

# Disable tqdm progress bars when not running in an interactive terminal
# (e.g. when called from Snakemake), to keep logs clean.
if not sys.stderr.isatty():
    os.environ.setdefault("TQDM_DISABLE", "1")

# libcudf means to default KvikIO to POSIX reads, but opens the first file
# before applying that, so each process probes cuFile/GPUDirect Storage once
# (seconds on nodes without it). Apply libcudf's intended default up front.
os.environ.setdefault("KVIKIO_COMPAT_MODE", "ON")

from . import io
from . import pp
from . import dr
from . import peak
from . import motif
from . import tobias
from . import pl
from . import gpu

__version__ = "0.1.0"
__all__ = ["io", "pp", "dr", "peak", "motif", "tobias", "pl", "gpu", "extensions", "set_verbosity"]


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
# A package extends GATAC by declaring an entry point in the "gatac.ext"
# group, e.g. in its pyproject.toml:
#
#     [project.entry-points."gatac.ext"]
#     chromafold = "gatac_chromafold"
#
# and ``gatac.chromafold`` then imports gatac_chromafold on first access.
# Core names always win: an extension cannot replace gatac.io, gatac.pp, ...


def extensions() -> dict[str, EntryPoint]:
    """
    Installed GATAC extensions, by the name they are reached at (``gatac.<name>``).

    Examples
    --------
    >>> import gatac as ga
    >>> ga.extensions()
    {'chromafold': EntryPoint(name='chromafold', value='gatac_chromafold', group='gatac.ext')}
    """
    return {ep.name: ep for ep in entry_points(group="gatac.ext")}


def __getattr__(name: str):
    ep = extensions().get(name) if not name.startswith("_") else None
    if ep is None:
        raise AttributeError(f"module 'gatac' has no attribute {name!r}")
    module = ep.load()
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(extensions()))


# ---------------------------------------------------------------------------
# Package-level logging
# ---------------------------------------------------------------------------
# Attach a single StreamHandler to the "gatac" root logger so that
# INFO messages from all submodules (gatac.dr.spectral, etc.) are visible
# when the library is imported interactively or in a script.
# Users can silence output with gatac.set_verbosity("WARNING") or by
# configuring their own handlers via the standard logging API.

_logger = logging.getLogger("gatac")
_logger.setLevel(logging.INFO)

_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                      datefmt="%H:%M:%S")
)
_logger.addHandler(_handler)
# Prevent messages from propagating to the root logger (avoids duplicate
# output when the caller also configures basicConfig).
_logger.propagate = False


def set_verbosity(level: str | int) -> None:
    """
    Set the logging verbosity for all GATAC submodules.

    Parameters
    ----------
    level
        A standard logging level name (``"DEBUG"``, ``"INFO"``,
        ``"WARNING"``, ``"ERROR"``) or the corresponding integer constant.

    Examples
    --------
    >>> import gatac
    >>> gatac.set_verbosity("DEBUG")   # show all messages
    >>> gatac.set_verbosity("WARNING") # silence info messages
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logging.getLogger("gatac").setLevel(level)


