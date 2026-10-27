"""gatac.io.Fasta, the genome reader behind the motif, chromVAR and TOBIAS functions."""
import gzip
import os

import numpy as np
import pytest

import gatac as ga
import gatac.io.fasta as fasta_mod

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


def test_gzipped_fasta_decompressed_once(tmp_path, monkeypatch):
    plain = _write(tmp_path / "g.fa")
    gz = tmp_path / "gz" / "g.fa.gz"
    gz.parent.mkdir()
    with open(plain, "rb") as src, gzip.open(gz, "wb") as dst:
        dst.write(src.read())
    calls = []
    real = fasta_mod._gunzip
    monkeypatch.setattr(fasta_mod, "_gunzip", lambda *a: (calls.append(a), real(*a)))
    for _ in range(3):
        with ga.io.Fasta(gz) as fa:
            assert fa.fetch(["chr1:2-30"]) == [SEQS["chr1"][2:30]]
    assert len(calls) == 1                                       # reopened from the same copy
    assert os.listdir(gz.parent) == ["g.fa.gz"]                  # nothing written next to the input
    os.utime(gz, ns=(0, 0))                                      # a changed file is decompressed again
    ga.io.Fasta(gz).close()
    assert len(calls) == 2


# FASTA layouts and their index as samtools faidx writes it; samtools leaves
# out an empty sequence, which is kept with length 0 as pyfaidx did
LAYOUTS = {
    "lf": (">a desc\nACGTA\nCGTAC\nGT\n>b\nAAAAA\n", "a\t12\t8\t5\t6\nb\t5\t26\t5\t6\n"),
    "crlf": (">a\r\nACGTA\r\nCGTAC\r\nGT\r\n>b x\r\nAAAAA\r\n", "a\t12\t4\t5\t7\nb\t5\t28\t5\t7\n"),
    "blank_lines_after": (">a\nACGTA\nCG\n\n\n>b\nAAAAA\n\n", "a\t7\t3\t5\t6\nb\t5\t17\t5\t6\n"),
    "no_final_newline": (">a\nACGTA\nCGTAC\n>b\nAAA", "a\t10\t3\t5\t6\nb\t3\t18\t3\t4\n"),
    "empty_sequence": (">a\nACGTA\nCG\n>e\n>b\nAAAAA\n", "a\t7\t3\t5\t6\ne\t0\t15\t0\t0\nb\t5\t18\t5\t6\n"),
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_index_matches_samtools(tmp_path, layout):
    text, fai = LAYOUTS[layout]
    (tmp_path / "g.fa").write_bytes(text.encode())
    fa = ga.io.Fasta(tmp_path / "g.fa")
    assert (tmp_path / "g.fa.fai").read_text() == fai
    seqs = {r.split("\n", 1)[0].split()[0]: r.split("\n", 1)[1].replace("\r", "").replace("\n", "")
            for r in text.split(">")[1:]}
    assert fa.fetch([f"{c}:0-100" for c in seqs]) == list(seqs.values())
    assert fa.fetch(["b:1-4"]) == [seqs["b"][1:4]]


def test_stale_index_is_rebuilt(tmp_path):
    path = _write(tmp_path / "g.fa")
    ga.io.Fasta(path).close()
    path.write_text(">chr9\nTTTT\n")
    os.utime(path.with_name("g.fa.fai"), (0, 0))                # index older than the FASTA
    with ga.io.Fasta(path) as fa:
        assert fa.chrom_sizes == {"chr9": 4}
    assert not list(tmp_path.glob("*.tmp"))


def test_index_rejects_ragged_lines(tmp_path):
    (tmp_path / "bad.fa").write_text(">a\nACGTA\nCG\nACGTA\n")
    with pytest.raises(ValueError, match="different lengths"):
        ga.io.Fasta(tmp_path / "bad.fa")


def test_index_in_memory_when_directory_is_read_only(tmp_path):
    d = tmp_path / "ro"
    d.mkdir()
    _write(d / "g.fa")
    d.chmod(0o555)
    try:
        with ga.io.Fasta(d / "g.fa") as fa:
            assert fa.fetch(["chr2:0-4"]) == ["GGGC"]
        assert not (d / "g.fa.fai").exists()
    finally:
        d.chmod(0o755)
