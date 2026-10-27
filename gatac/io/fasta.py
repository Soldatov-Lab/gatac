"""
Genome FASTA access shared by every GATAC function that reads sequence.

Sequence is read straight from the file through its ``.fai`` index (built by
pyfaidx when missing), with no per-region Python object: a region is a set of
byte offsets into a memory map. Gzipped FASTA files are decompressed once to a
temporary file, which is removed when the :class:`Fasta` is closed.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


class Fasta:
    """
    Random access to a genome FASTA file.

    Parameters
    ----------
    path
        FASTA file (``.fa``, ``.fasta``, optionally gzipped). The ``.fai`` index
        next to it is used, or built if missing; a gzipped file is decompressed
        to a temporary directory first.

    Examples
    --------
    >>> import gatac as ga
    >>> with ga.io.Fasta("GRCh38.fa") as fa:
    ...     fa.chrom_sizes["chr19"]
    ...     fa.fetch(["chr19:1000000-1000010"])
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._tmpdir = None
        fasta = self.path
        if fasta.name.endswith(".gz"):
            self._tmpdir = tempfile.mkdtemp(prefix="gatac_fasta_")
            fasta = Path(self._tmpdir) / fasta.name[:-3]
            _gunzip(self.path, fasta)
        fai = Path(str(fasta) + ".fai")
        if not fai.exists():
            from pyfaidx import Fasta as _Faidx

            _Faidx(str(fasta))
        #: ``{chrom: (length, byte offset, bases per line, bytes per line)}``
        self.index = {}
        for line in open(fai):
            f = line.split("\t")
            self.index[f[0]] = (int(f[1]), int(f[2]), int(f[3]), int(f[4]))
        self._file = fasta
        self._mm = np.memmap(fasta, dtype=np.uint8, mode="r")

    def __repr__(self) -> str:
        return f"Fasta({str(self.path)!r}, {len(self.index)} sequences)"

    def __enter__(self) -> "Fasta":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __contains__(self, chrom: str) -> bool:
        return chrom in self.index

    def close(self) -> None:
        """Release the memory map and remove the decompressed copy, if any."""
        self._mm = None
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    @property
    def chrom_sizes(self) -> dict[str, int]:
        """Length of every sequence, in file order."""
        return {c: v[0] for c, v in self.index.items()}

    def chrom(self, chrom: str) -> np.ndarray:
        """Bytes of one whole sequence (uint8, newlines removed, case kept)."""
        length, off, lb, lw = self.index[chrom]
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
            # one contiguous block of the file, line ends removed
            seq = self._mm[off + (s // lb) * lw + s % lb:off + (e // lb) * lw + e % lb].tobytes()
            if lw != lb:
                seq = seq.replace(b"\n", b"").replace(b"\r", b"")
            out.append(seq.decode("ascii"))
        return out


def _gunzip(src: Path, dst: Path) -> None:
    """Decompress ``src`` to ``dst``, in parallel with rapidgzip when installed."""
    logger.info(f"Decompressing {src.name}...")
    try:
        import rapidgzip

        opener = rapidgzip.open
    except ImportError:
        import gzip

        opener = gzip.open
    with opener(str(src)) as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, 64 * 1024 * 1024)
