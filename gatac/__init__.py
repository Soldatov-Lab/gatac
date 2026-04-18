"""
GATAC - GPU-accelerated ATAC-seq processing toolkit.

Public API:
    - pp: Preprocessing functions
    - tl: Tools (analysis functions)
"""

import logging
import os
import sys

# Disable tqdm progress bars when not running in an interactive terminal
# (e.g. when called from Snakemake), to keep logs clean.
if not sys.stderr.isatty():
    os.environ.setdefault("TQDM_DISABLE", "1")

from . import pp
from . import tl
from . import pl

__version__ = "0.1.0"
__all__ = ["pp", "tl", "pl", "set_verbosity"]

# ---------------------------------------------------------------------------
# Package-level logging
# ---------------------------------------------------------------------------
# Attach a single StreamHandler to the "gatac" root logger so that
# INFO messages from all submodules (gatac.tl.spectral, etc.) are visible
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


