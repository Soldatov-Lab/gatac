"""gatac.io.Fasta, the genome reader behind the motif, chromVAR and TOBIAS functions."""
import gzip
import os

import numpy as np
import pytest

import gatac as ga

SEQS = {"chr1": "ACGTacgtNNACGTTTGCA" * 7, "chr2": "GGGCCCAAAT", "chrM": "acgtn" * 3}


def _write(path, width=7, newline="\n"):
    with open(path, "w", newline="") as fh:
        for c, s in SEQS.items():
            fh.write(f">{c} some description{newline}")
            fh.write("".join(s[i:i + width] + newline for i in range(0, len(s), width)))
    return path


@pytest.mark.parametrize("width,newline", [(7, "\n"), (60, "\n"), (7, "\r\n")])
def test_fetch_matches_slicing(tmp_path, width, newline):
    with ga.io.Fasta(_write(tmp_path / "g.fa", width, newline)) as fa:
        assert fa.chrom_sizes == {c: len(s) for c, s in SEQS.items()}
        regions = [f"{c}:{s}-{e}" for c, seq in SEQS.items()
                   for s in range(0, len(seq), 3) for e in (s, s + 1, s + 8, len(seq))]
        assert fa.fetch(regions) == [SEQS[r.split(":")[0]][int(r.split(":")[1].split("-")[0]):int(r.split("-")[1])]
                                     for r in regions]
        for c, s in SEQS.items():
            assert fa.chrom(c).tobytes().decode() == s
        got = fa.region_bytes("chr1", np.array([0, 5, 30]), np.array([3, 12, 31])).tobytes().decode()
        assert got == SEQS["chr1"][0:3] + SEQS["chr1"][5:12] + SEQS["chr1"][30:31]


def test_fetch_matches_pyfaidx(tmp_path):
    from pyfaidx import Fasta as Faidx

    path = _write(tmp_path / "g.fa")
    ref = Faidx(str(path), one_based_attributes=False)
    regions = ["chr1:3-40", "chr2:0-10", "chr2:5-500", "chrM:2-9", "chr1:50-40"]
    with ga.io.Fasta(path) as fa:
        assert fa.fetch(regions) == [str(ref[r.split(":")[0]][int(r.split(":")[1].split("-")[0]):int(r.split("-")[1])])
                                     for r in regions]


def test_fetch_one_based_and_bad_regions(tmp_path, caplog):
    with ga.io.Fasta(_write(tmp_path / "g.fa")) as fa:
        assert fa.fetch(["chr2:1-3", "chr2:8-10"], one_based=True) == ["GGG", "AAT"]
        assert fa.fetch(["chr2:0-3"], one_based=True) == ["GGG"]           # start clipped to 0
        assert fa.fetch(["chrNope:1-5", "garbage", "chr2:x-5", "chr2:1-3"]) == ["", "", "", "GG"]
    assert caplog.text.count("Failed to fetch sequence") == 3
    assert "chrNope not in" in caplog.text


def test_gzipped_fasta(tmp_path):
    plain = _write(tmp_path / "g.fa")
    gz = tmp_path / "gz" / "g.fa.gz"
    gz.parent.mkdir()
    with open(plain, "rb") as src, gzip.open(gz, "wb") as dst:
        dst.write(src.read())
    fa = ga.io.Fasta(gz)
    tmp = fa._tmpdir
    assert fa.fetch(["chr1:2-30"]) == [SEQS["chr1"][2:30]]
    fa.close()
    assert not os.path.exists(tmp)
    assert os.listdir(gz.parent) == ["g.fa.gz"]                  # nothing written next to the input
