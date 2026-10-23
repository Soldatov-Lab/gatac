"""The shared metrics, BED and GTF readers, and the functions that use them."""
import gzip
import re

import cudf
import numpy as np
import pandas as pd
import pytest

import gatac as ga
from gatac._readers import read_bed, read_gtf, read_metrics


def _chrom_sizes(world):
    from pyfaidx import Fasta
    return {c: len(seq) for c, seq in Fasta(str(world["fasta"])).items()}


METRICS = pd.DataFrame({"barcode": ["AAA-1", "CCC-1"], "n_unique": [1500, 200], "tsse_score": [7.5, 2.0]})


@pytest.mark.parametrize("kind", ["csv", "str", "pandas", "cudf", "polars"])
def test_read_metrics_inputs(kind, tmp_path):
    path = tmp_path / "metrics.csv"
    METRICS.to_csv(path, index=False)
    metrics = {
        "csv": lambda: path,
        "str": lambda: str(path),
        "pandas": lambda: METRICS,
        "cudf": lambda: cudf.from_pandas(METRICS),
        "polars": lambda: pytest.importorskip("polars").from_pandas(METRICS),
    }[kind]()
    got = read_metrics(metrics)
    assert isinstance(got, pd.DataFrame)
    pd.testing.assert_frame_equal(got, METRICS)
    assert isinstance(read_metrics(metrics, gpu=True), cudf.DataFrame)


def test_read_metrics_copies_a_dataframe():
    got = read_metrics(METRICS)
    got.loc[0, "n_unique"] = 0
    assert METRICS.loc[0, "n_unique"] == 1500


def test_read_metrics_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_metrics(tmp_path / "typo.csv")


def test_read_metrics_rejects_other_types():
    with pytest.raises(TypeError):
        read_metrics(["AAA-1"])


def test_make_tile_matrix_missing_metrics_raises(world, tmp_path):
    """A mistyped metrics path must not silently skip the cell filter."""
    with pytest.raises(FileNotFoundError, match="Metrics file not found"):
        ga.pp.make_tile_matrix(world["parquet"], chrom_sizes=_chrom_sizes(world),
                               metrics=tmp_path / "typo.csv", filter_query="n_unique >= 100")


@pytest.mark.parametrize("kind", ["pandas", "polars", "csv"])
def test_make_tile_matrix_metrics_inputs(kind, world, tmp_path):
    """Every metrics input gives the cells of a cuDF one."""
    counts = ga.io.FragmentFile(world["parquet"]).barcode_counts()
    metrics = counts.assign(tsse_score=np.arange(len(counts)) % 10)
    path = tmp_path / "metrics.csv"
    metrics.to_csv(path, index=False)
    other = {"pandas": lambda: metrics, "csv": lambda: path,
             "polars": lambda: pytest.importorskip("polars").from_pandas(metrics)}[kind]()
    kw = dict(chrom_sizes=_chrom_sizes(world), tile_size=5_000, filter_query="tsse_score >= 5")
    ref = ga.pp.make_tile_matrix(world["parquet"], metrics=cudf.from_pandas(metrics), **kw)
    got = ga.pp.make_tile_matrix(world["parquet"], metrics=other, **kw)
    assert 0 < got.n_obs < len(counts)
    assert got.obs_names.equals(ref.obs_names)
    assert (got.X != ref.X).nnz == 0


@pytest.mark.parametrize("gz", [False, True])
def test_read_bed(gz, tmp_path):
    text = ("track name=peaks\n"
            "browser position chr1:1-100\n"
            "# a comment\n"
            "1\t100\t200\tpeak1\t5\n"
            "chrX\t300\t450\tpeak2\t7\n")
    path = tmp_path / ("peaks.bed.gz" if gz else "peaks.bed")
    if gz:
        with gzip.open(path, "wt") as f:
            f.write(text)
    else:
        path.write_text(text)
    got = read_bed(path)
    assert list(got.columns) == ["chrom", "start", "end"]
    assert got["chrom"].tolist() == ["1", "chrX"]       # numeric names stay strings
    assert got["start"].tolist() == [100, 300] and got["end"].tolist() == [200, 450]
    assert got["start"].dtype == np.int64 and got["end"].dtype == np.int64


GTF = (
    "##description: test\n"
    "#!genome-build GRCh38\n"
    'chr1\tHAVANA\tgene\t100\t900\t.\t+\t.\tgene_id "G1"; gene_name "A";\n'
    'chr1\tHAVANA\ttranscript\t100\t500\t.\t+\t.\tgene_id "G1"; transcript_id "T1"; gene_name "A";\n'
    'chr1\tHAVANA\ttranscript\t300\t900\t.\t+\t.\tgene_id "G1"; transcript_id "T2";\n'
    'chr2\tHAVANA\ttranscript\t50\t80\t.\t-\t.\tgene_id "G2"; transcript_id "T3"; gene_name "B";\n'
)


@pytest.mark.parametrize("fmt", ["gtf", "gtf.gz", "gff3"])
def test_read_gtf(fmt, tmp_path):
    text = GTF
    if fmt == "gff3":   # same records with key=value; attributes
        text = re.sub(r'(\w+) "([^"]*)"; ?', r"\1=\2;", text)
    path = tmp_path / f"genes.{fmt}"
    if fmt.endswith(".gz"):
        with gzip.open(path, "wt") as f:
            f.write(text)
    else:
        path.write_text(text)
    got = read_gtf(path, "transcript", {"name": "gene_name", "tid": "transcript_id"})
    assert list(got.columns) == ["chrom", "start", "end", "strand", "name", "tid"]
    assert got["tid"].tolist() == ["T1", "T2", "T3"]                  # file order
    assert got["start"].tolist() == [100, 300, 50] and got["end"].tolist() == [500, 900, 80]
    assert got["name"].tolist()[0::2] == ["A", "B"] and pd.isna(got["name"][1])   # absent key: null
    assert got["start"].dtype == np.int64


def test_bindetect_site_order_with_underscore_chroms():
    """Sites ordered by motif, then by TOBIAS's "chrom_start_end" id in Python string order."""
    from types import SimpleNamespace

    from gatac.tobias.bindetect import _tobias_site_order
    rng = np.random.default_rng(0)
    chroms = ["chr1", "chr1_KI270706v1_random", "chr10", "chr2", "chrUn_GL000220v1"]
    n = 5_000
    sites = SimpleNamespace(
        chroms=chroms, motif=rng.integers(0, 4, n).astype(np.int32),
        chrom=rng.integers(0, len(chroms), n).astype(np.int16),
        start=rng.integers(0, 3_000, n).astype(np.int64))
    sites.end = sites.start + rng.integers(5, 30, n)
    sites.start[:200] = sites.start[200:400]                 # shared starts and full ties
    sites.end[:100], sites.chrom[:100] = sites.end[200:300], sites.chrom[200:300]

    def ids(idx):
        return [(sites.motif[i], f"{chroms[sites.chrom[i]]}_{sites.start[i]}_{sites.end[i]}") for i in idx]

    expect = sorted(range(n), key=lambda i: ids([i])[0])    # sorted() is stable
    assert _tobias_site_order(sites).tolist() == expect
    sub = np.sort(rng.choice(n, 1_000, replace=False))
    assert _tobias_site_order(sites, sub).tolist() == sorted(range(len(sub)), key=lambda j: ids([sub[j]])[0])
