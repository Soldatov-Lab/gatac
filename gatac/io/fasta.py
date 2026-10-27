"""
Genome FASTA access shared by every GATAC function that reads sequence.

Sequence is read straight from the file through its ``.fai`` index, with no
per-region Python object: a region is a set of byte offsets into a memory map.
A missing index is built with NumPy and written next to the file when the
directory is writable.

A gzipped FASTA file is decompressed once per process, to a temporary
directory removed when the process exits: every later :class:`Fasta` of the
same file (same path, size and modification time) reuses that copy and its
index, so functions that open the genome several times, or several
functions in a row, do not decompress it again.
"""
from __future__ import annotations

import atexit
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

# Bytes scanned at a time when building an index
_CHUNK = 1 << 26

# Decompressed copies of gzipped FASTA files, for the life of the process:
# (resolved path, size, mtime) -> (plain copy, its index)
_PLAIN_COPIES: dict = {}
_PLAIN_LOCK = threading.Lock()
_PLAIN_DIR = None


class Fasta:
    """
    Random access to a genome FASTA file.

    Parameters
    ----------
    path
        FASTA file (``.fa``, ``.fasta``, optionally gzipped). The ``.fai`` index
        next to it is used, or built if missing; a gzipped file is decompressed
        to a temporary directory, once per process.

    Attributes
    ----------
    index : dict
        ``{chrom: (length, byte offset, bases per line, bytes per line)}``, as
        in a ``.fai`` file.

    Examples
    --------
    >>> import gatac as ga
    >>> with ga.io.Fasta("GRCh38.fa") as fa:
    ...     fa.chrom_sizes["chr19"]
    ...     fa.fetch(["chr19:1000000-1000010"])
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if self.path.name.endswith(".gz"):
            fasta, self.index = _plain_copy(self.path)
        else:
            fasta, self.index = self.path, _load_index(self.path)
        self._mm = np.memmap(fasta, dtype=np.uint8, mode="r")

    def __repr__(self) -> str:
        return f"Fasta({str(self.path)!r}, {len(self.index)} sequences)"

    def __enter__(self) -> "Fasta":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __contains__(self, chrom: str) -> bool:
        return chrom in self.index

    def close(self) -> None:
        """Release the memory map (a decompressed copy stays for the process)."""
        self._mm = None

    @property
    def chrom_sizes(self) -> dict[str, int]:
        """Length of every sequence, in file order."""
        return {c: v[0] for c, v in self.index.items()}

    def chrom(self, chrom: str) -> np.ndarray:
        """Bytes of one whole sequence (uint8, newlines removed, case kept)."""
        length, off, lb, lw = self.index[chrom]
        if length == 0:
            return np.zeros(0, dtype=np.uint8)
        n_bytes = (length // lb) * lw + length % lb
        blk = np.asarray(self._mm[off:off + n_bytes])
        if lw == lb:
            return blk.copy()
        return blk[(blk != 10) & (blk != 13)][:length]

    def region_bytes(self, chrom: str, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Bytes of the 0-based, half-open intervals ``[start, end)`` of one sequence,
        concatenated (uint8, case kept). Intervals must lie within the sequence."""
        _, off, lb, lw = self.index[chrom]
        starts = np.asarray(starts, dtype=np.int64)
        lens = np.asarray(ends, dtype=np.int64) - starts
        within = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        gpos = np.repeat(starts, lens) + within
        return np.asarray(self._mm[off + (gpos // lb) * lw + gpos % lb])

    def fetch(self, regions: Sequence[str], one_based: bool = False) -> list[str]:
        """
        Sequences of ``chrom:start-end`` regions, case kept.

        Parameters
        ----------
        regions
            Region names, e.g. ``adata.var_names`` of a peak matrix.
        one_based
            Regions are 1-based and closed (R/GenomicRanges) rather than 0-based
            and half-open (BED).

        Returns
        -------
        One string per region, clipped to the sequence end. Regions that cannot
        be parsed or are on an unknown sequence give ``""`` with a warning.
        """
        out = []
        for r in regions:
            try:
                c, coords = r.rsplit(":", 1)
                s, e = coords.split("-")
                s, e = int(s) - one_based, int(e)
                length, off, lb, lw = self.index[c]
            except (ValueError, KeyError) as err:
                if isinstance(err, KeyError):
                    err = f"{err.args[0]} not in {self.path}"
                logger.warning(f"Failed to fetch sequence for {r}: {err}")
                out.append("")
                continue
            s = min(max(s, 0), length)
            e = min(max(e, s), length)
            if e == s:
                out.append("")
                continue
            # one contiguous block of the file, line ends removed
            seq = self._mm[off + (s // lb) * lw + s % lb:off + (e // lb) * lw + e % lb].tobytes()
            if lw != lb:
                seq = seq.replace(b"\n", b"").replace(b"\r", b"")
            out.append(seq.decode("ascii"))
        return out


def _plain_copy(path: Path) -> tuple[Path, dict]:
    """Decompressed copy of a gzipped FASTA and its index, made once per process."""
    global _PLAIN_DIR
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    with _PLAIN_LOCK:
        if key not in _PLAIN_COPIES:
            if _PLAIN_DIR is None:
                _PLAIN_DIR = tempfile.mkdtemp(prefix="gatac_fasta_")
                atexit.register(shutil.rmtree, _PLAIN_DIR, True)
            dst = Path(_PLAIN_DIR) / f"{len(_PLAIN_COPIES)}_{path.name[:-3]}"
            _gunzip(path, dst)
            _PLAIN_COPIES[key] = (dst, _load_index(dst))
        return _PLAIN_COPIES[key]


def _gunzip(src: Path, dst: Path) -> None:
    """Decompress ``src`` to ``dst``, in parallel with rapidgzip when installed."""
    logger.info(f"Decompressing {src.name} (once per process)...")
    try:
        import rapidgzip

        opener = rapidgzip.open
    except ImportError:
        import gzip

        opener = gzip.open
    with opener(str(src)) as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, 64 * 1024 * 1024)


def _load_index(fasta: Path) -> dict[str, tuple[int, int, int, int]]:
    """The ``.fai`` index of ``fasta``: read, or built and written when missing or
    older than the FASTA file."""
    fai = Path(str(fasta) + ".fai")
    if fai.exists() and fai.stat().st_mtime >= fasta.stat().st_mtime:
        index = {}
        for line in open(fai):
            f = line.split("\t")
            index[f[0]] = (int(f[1]), int(f[2]), int(f[3]), int(f[4]))
        return index
    index = _build_index(fasta)
    tmp = fai.with_name(f"{fai.name}.{os.getpid()}.tmp")       # readers never see a partial index
    try:
        with open(tmp, "w") as f:
            f.writelines(f"{c}\t{n}\t{o}\t{lb}\t{lw}\n" for c, (n, o, lb, lw) in index.items())
        os.replace(tmp, fai)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        logger.info(f"Could not write {fai} ({e}); the index is kept in memory")
    return index


def _build_index(fasta: Path) -> dict[str, tuple[int, int, int, int]]:
    """``samtools faidx`` index of a FASTA file, from the positions of its line ends."""
    logger.info(f"Indexing {fasta.name}...")
    mm = np.memmap(fasta, dtype=np.uint8, mode="r")
    n = len(mm)
    nl = np.concatenate([np.flatnonzero(mm[i:i + _CHUNK] == 10) + i for i in range(0, n, _CHUNK)]
                        + [np.zeros(0, np.int64)])
    # line k is [ls[k], le[k]), without its "\n"
    ls = np.concatenate([[0], nl + 1])
    le = np.append(nl, n)
    del nl
    if ls[-1] == n:
        ls, le = ls[:-1], le[:-1]
    cr = (le > ls) & (mm[np.maximum(le - 1, 0)] == 13)
    bases = le - ls - cr
    header = np.flatnonzero((le > ls) & (mm[np.minimum(ls, n - 1)] == ord(">")))
    index = {}
    for k, h in enumerate(header):
        name = bytes(mm[ls[h] + 1:le[h] - cr[h]]).decode().split()[0]
        if name in index:
            raise ValueError(f"{fasta}: duplicate sequence name {name}")
        a = h + 1
        b = header[k + 1] if k + 1 < len(header) else len(ls)
        filled = np.flatnonzero(bases[a:b])
        if len(filled) == 0:
            index[name] = (0, int(ls[a]) if a < len(ls) else n, 0, 0)
            continue
        b = a + filled[-1] + 1                       # trailing blank lines are not sequence
        lb = int(bases[a])
        lw = int(ls[a + 1] - ls[a]) if b - a > 1 else lb + 1 + int(cr[a])
        if (bases[a:b - 1] != lb).any() or bases[b - 1] > lb or (np.diff(ls[a:b]) != lw).any():
            raise ValueError(f"{fasta}: lines of {name} have different lengths; it cannot be indexed")
        index[name] = (int(bases[a:b].sum()), int(ls[a]), lb, lw)
    return index
