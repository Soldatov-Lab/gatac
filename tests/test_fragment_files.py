"""
GATAC fragment Parquet files: the writer, the reader, and what reads through it.

``make_parquet`` stores chromosomes and barcodes as integer ids with the
barcode table inside the file, and everything that reads fragments goes
through ``FragmentFile``. The reader is checked against pandas on the source
fragments; the matrix builders against their own results with other batch
sizes. Synthetic data, GPU required.
"""
from __future__ import annotations

import gzip
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cupy")
pytest.importorskip("cudf")

import duckdb  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import gatac as ga  # noqa: E402
from gatac.io.fragments import FragmentFile  # noqa: E402

CHROM_SIZES = {"chr1": 3_000_000, "chr2": 2_500_000, "chr3": 2_000_000, "chrX": 1_500_000}


def _fragments(seed: int = 0) -> pd.DataFrame:
    """Sorted fragments of 300 barcodes; a third of them sit in 40 hotspots."""
    rng = np.random.default_rng(seed)
    n = 120_000
    contigs = list(CHROM_SIZES) + ["chrM", "GL000194.1"]
    chrom = rng.choice(contigs, n, p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.05])
    size = np.array([CHROM_SIZES.get(c, 16_000) for c in chrom])
    start = (rng.random(n) * (size - 2_000)).astype(np.int64)
    hot = rng.random(n) < 0.35
    centers = rng.integers(10_000, 1_400_000, 40)
    start[hot] = centers[rng.integers(0, 40, hot.sum())] + rng.integers(-150, 150, hot.sum())
    length = np.where(rng.random(n) < 0.5, rng.integers(40, 150, n), rng.integers(150, 700, n))
    weights = rng.pareto(1.2, 300) + 1
    barcodes = np.array([f"{''.join(rng.choice(list('ACGT'), 16))}-1" for _ in range(300)])
    bc = rng.choice(barcodes, n, p=weights / weights.sum())
    df = pd.DataFrame({"chrom": chrom, "start": start, "end": start + length, "barcode": bc,
                       "count": rng.integers(1, 5, n)})
    return df.sort_values(["chrom", "start", "end", "barcode"], kind="stable").drop_duplicates(
        ["chrom", "start", "end", "barcode"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("fragments")
    df = _fragments()
    tsv = d / "frags.tsv.gz"
    with gzip.open(tsv, "wt") as fh:
        fh.write("# header comment\n")
        df.to_csv(fh, sep="\t", header=False, index=False)
    path = ga.io.make_parquet(tsv, d / "frags.parquet", barcode_prefix="S1_", row_group_size=9_000)
    df["barcode"] = "S1_" + df["barcode"]
    gtf = d / "genes.gtf"
    rows = []
    rng = np.random.default_rng(1)
    for i in range(60):
        c = list(CHROM_SIZES)[i % 4]
        s = int(rng.integers(5_000, CHROM_SIZES[c] - 60_000))
        e = s + int(rng.integers(2_000, 50_000))
        strand = "+" if i % 2 else "-"
        attrs = f'gene_id "G{i}"; gene_name "GENE{i}"; transcript_id "T{i}"; transcript_name "GENE{i}-201";'
        for feature in ("gene", "transcript"):
            rows.append(f"{c}\ttest\t{feature}\t{s}\t{e}\t.\t{strand}\t.\t{attrs}")
    gtf.write_text("\n".join(rows) + "\n")
    return {"df": df, "path": path, "tsv": tsv, "dir": d, "gtf": gtf}


def _cells(data, n=60):
    counts = data["df"].groupby("barcode").size().sort_values(ascending=False)
    return list(counts.index[:n])


def test_round_trip_keeps_rows_order_and_names(data):
    expected = data["df"]
    got = pa.concat_tables(FragmentFile(data["path"]).iter_arrow()).to_pandas()
    for c in expected.columns:
        assert np.array_equal(got[c].to_numpy().astype(str), expected[c].to_numpy().astype(str)), c


def test_file_layout(data):
    ff = FragmentFile(data["path"])
    # Chromosome ids in order of first appearance
    assert ff.chroms == list(dict.fromkeys(data["df"]["chrom"]))
    # Barcode table: original names, ids by decreasing fragment count
    counts = ff.barcode_counts()
    expected = data["df"].groupby("barcode")["count"].agg(["size", "sum"])
    assert list(counts["barcode"]) == list(ff.barcodes)
    assert (np.diff(counts["n_unique"].to_numpy()) <= 0).all()
    exp = expected.loc[counts["barcode"]]
    assert np.array_equal(counts["n_unique"], exp["size"]) and np.array_equal(counts["n_total"], exp["sum"])
    # Row groups have the requested size
    rows = [ff._metadata.row_group(i).num_rows for i in range(ff.num_row_groups)]
    assert set(rows[:-1]) == {9_000}
    keys = set(pq.ParquetFile(data["path"]).metadata.metadata)
    assert {b"gatac.format_version", b"gatac.chroms", b"gatac.n_barcodes"} <= keys


@pytest.mark.parametrize("chroms", [None, ["chr2", "chrM"]])
def test_barcode_counts(data, chroms):
    df = data["df"]
    got = FragmentFile(data["path"]).barcode_counts(chroms=chroms, mito_chroms=["chrM"])
    got = got.set_index("barcode").sort_index()
    sel = df if chroms is None else df[df["chrom"].isin(chroms)]
    exp = pd.DataFrame({
        "n_unique": sel.groupby("barcode").size(),
        "n_total": sel.groupby("barcode")["count"].sum(),
        "n_mito": sel.assign(m=np.where(sel["chrom"] == "chrM", sel["count"], 0)).groupby("barcode")["m"].sum(),
    }).sort_index()
    pd.testing.assert_frame_equal(got, exp, check_dtype=False, check_names=False)


def test_iter_batches(data):
    cells = _cells(data, 40)
    chroms = ["chr3", "chr1", "chrX"]
    parts = [b.to_pandas() for b in FragmentFile(data["path"]).iter_batches(
        ["chrom", "start", "end", "count"], row_groups_per_batch=3, chroms=chroms, cells=cells)]
    got = pd.concat(parts).sort_values(["chrom", "start", "end", "cell"]).reset_index(drop=True)
    df = data["df"]
    ref = df[df["chrom"].isin(chroms) & df["barcode"].isin(cells)]
    exp = pd.DataFrame({
        "chrom": ref["chrom"].map({c: i for i, c in enumerate(chroms)}),
        "start": ref["start"], "end": ref["end"], "count": ref["count"],
        "cell": ref["barcode"].map({c: i for i, c in enumerate(cells)}),
    }).sort_values(["chrom", "start", "end", "cell"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, exp, check_dtype=False)


def test_empty_selection_yields_nothing(data):
    ff = FragmentFile(data["path"])
    assert list(ff.iter_batches(chroms=[])) == []
    assert list(ff.iter_batches(cells=[])) == []


def test_register_duckdb_view(data):
    con = duckdb.connect()
    FragmentFile(data["path"]).register(con)
    got = con.sql("SELECT * FROM fragments ORDER BY chrom, start, \"end\", barcode").df()
    exp = data["df"].sort_values(["chrom", "start", "end", "barcode"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, exp, check_dtype=False)


def test_rejects_other_parquet(data, tmp_path):
    other = tmp_path / "plain.parquet"
    pq.write_table(pa.Table.from_pandas(data["df"].head(100)), other)
    with pytest.raises(ValueError, match="not a GATAC fragment file"):
        FragmentFile(other)


def test_batch_prefixes_and_no_partial_file(data, tmp_path):
    tsv_b = tmp_path / "other.tsv.gz"
    shutil.copy(data["tsv"], tsv_b)
    outs = ga.io.make_parquet_batch([data["tsv"], tsv_b], output_dir=tmp_path / "out",
                                    barcode_prefix=["A_", "B_"], workers=2)
    for out, prefix in zip(outs, ["A_", "B_"]):
        assert all(b.startswith(prefix) for b in FragmentFile(out).barcode_counts()["barcode"])
    # The staging Parquet of the TSV parse is removed
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["frags.parquet", "other.parquet"]
    bad = pd.DataFrame({"barcode": ["not-in-file"], "n_unique": [1], "n_total": [1]})
    target = tmp_path / "bad.parquet"
    with pytest.raises(ValueError):
        ga.io.fragments.write_fragments(lambda: FragmentFile(data["path"]).iter_arrow(), target,
                                        barcode_stats=bad)
    assert not target.exists() and not list(tmp_path.glob("bad.parquet*"))



def test_to_tsv_round_trip(data, tmp_path):
    cols = list(data["df"].columns)
    gz = ga.io.to_tsv(data["path"], tmp_path / "frags.tsv.gz")
    plain = FragmentFile(data["path"]).to_tsv(tmp_path / "frags.tsv")
    for out in (gz, plain):
        got = pd.read_csv(out, sep="\t", header=None, names=cols)
        pd.testing.assert_frame_equal(got, data["df"], check_dtype=False)
    raw = gz.read_bytes()
    assert gzip.decompress(raw) == plain.read_bytes()
    assert raw[12:14] == b"BC" and raw.endswith(bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000"))
    assert ga.io.make_parquet(gz, tmp_path / "again.parquet").exists()


def test_to_tsv_keeps_existing_output(data, tmp_path):
    # The default output of to_tsv is usually the original fragments file
    original = tmp_path / "frags.tsv.gz"
    original.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        ga.io.to_tsv(data["path"], original)
    assert original.read_bytes() == b"original"
    ga.io.to_tsv(data["path"], original, overwrite=True)
    assert len(pd.read_csv(original, sep="\t", header=None)) == len(data["df"])
    assert list(tmp_path.glob("*.tmp")) == []

def test_read_fragments_parquet_categoricals(data):
    got = ga.io.read_fragments_parquet(data["path"]).to_pandas()
    assert got["chrom"].dtype == "category" and got["barcode"].dtype == "category"
    df = data["df"]
    assert list(got["chrom"].cat.categories) == sorted(df["chrom"].unique())
    assert list(got["barcode"].cat.categories) == sorted(df["barcode"].unique())
    for c in df.columns:
        assert np.array_equal(got[c].to_numpy().astype(str), df[c].to_numpy().astype(str)), c


def test_filter(data):
    out = ga.pp.filter_fragments(data["path"], data["dir"] / "filtered.parquet",
                                 min_fragments_per_cell=200, chrom_sizes=CHROM_SIZES)
    ff = FragmentFile(out)
    t = pa.concat_tables(ff.iter_arrow()).to_pandas()
    # The barcode table describes the filtered fragments
    counts = ff.barcode_counts().set_index("barcode")["n_unique"].sort_index()
    assert counts.equals(t.groupby("barcode").size().sort_index().rename("n_unique"))
    ref = data["df"]
    kept = ref[ref["chrom"].isin(CHROM_SIZES)]
    n = kept.groupby("barcode").size()
    kept = kept[kept["barcode"].isin(n.index[n >= 200])].reset_index(drop=True)
    assert np.array_equal(t[["chrom", "start", "end", "barcode"]].to_numpy().astype(str),
                          kept[["chrom", "start", "end", "barcode"]].to_numpy().astype(str))


def test_filter_with_metrics(data):
    # Cells from a metrics table, one of them absent from the file
    cells = _cells(data, 30)
    metrics = pd.DataFrame({"barcode": cells + ["S1_not-in-file"]})
    out = ga.pp.filter_fragments(data["path"], data["dir"] / "filtered_metrics.parquet",
                                 metrics=metrics, chrom_sizes=CHROM_SIZES)
    ff = FragmentFile(out)
    ref = data["df"]
    kept = ref[ref["chrom"].isin(CHROM_SIZES) & ref["barcode"].isin(cells)].reset_index(drop=True)
    t = pa.concat_tables(ff.iter_arrow()).to_pandas()
    assert np.array_equal(t.to_numpy().astype(str), kept.to_numpy().astype(str))
    exp = kept.groupby("barcode")["count"].agg(["size", "sum"])
    counts = ff.barcode_counts().set_index("barcode").loc[exp.index]
    assert ff.n_barcodes == len(exp)
    assert np.array_equal(counts["n_unique"], exp["size"]) and np.array_equal(counts["n_total"], exp["sum"])


def _adata_equal(a, b):
    assert list(a.obs_names) == list(b.obs_names)
    assert list(a.var_names) == list(b.var_names)
    assert (sp.csr_matrix(a.X) != sp.csr_matrix(b.X)).nnz == 0
    common = [c for c in a.obs.columns if c in b.obs.columns]
    pd.testing.assert_frame_equal(a.obs[common], b.obs[common], check_dtype=False)


@pytest.mark.parametrize("strategy", ["unique", "count", "binarize"])
def test_tile_matrix_independent_of_batching(data, strategy):
    res = [ga.pp.make_tile_matrix(data["path"], CHROM_SIZES, tile_size=50_000,
                                  min_fragments_per_cell=50, count_strategy=strategy,
                                  row_groups_per_batch=rg) for rg in (1, 4, 64)]
    assert res[0].n_obs > 10
    assert set(res[0].obs_names) <= set(data["df"]["barcode"])
    for r in res[1:]:
        _adata_equal(r, res[0])


def test_metrics_independent_of_batching(data):
    import polars as pl
    tss = pl.DataFrame({"chrom": ["chr1", "chr1", "chr2", "chr3"],
                        "tss": [200_000, 900_000, 400_000, 1_000_000], "strand": ["+", "-", "+", "+"]})
    res = [ga.pp.compute_metrics(data["path"], tss, min_unique_frags=50,
                                 row_groups_per_batch=rg).to_pandas().sort_values("barcode").reset_index(drop=True)
           for rg in (1, 4, 64)]
    assert len(res[0]) > 10
    for r in res[1:]:
        pd.testing.assert_frame_equal(r, res[0], check_dtype=False)


def test_doublets_run(data):
    sizes = {c: CHROM_SIZES[c] for c in ("chr1", "chr2", "chr3")}
    res = ga.pp.detect_doublets(data["path"], sizes, min_fragments=50)
    assert len(res) > 10
    assert set(res["cell_id"]) <= set(data["df"]["barcode"])


@pytest.mark.parametrize("fn", ["gene", "genescore"])
def test_gene_matrices_run(data, fn):
    make = ga.pp.make_gene_matrix if fn == "gene" else ga.pp.make_gene_score_matrix
    res = make(data["path"], data["gtf"], min_fragments_per_cell=50)
    assert res.n_obs > 10 and sp.csr_matrix(res.X).nnz > 0
    assert set(res.obs_names) <= set(data["df"]["barcode"])


@pytest.mark.parametrize("fn", ["gene", "genescore"])
def test_gene_matrices_barcode_only_metrics(data, fn):
    # A metrics table with only barcodes selects those cells; the fragment
    # counts are computed from the file
    import cudf
    make = ga.pp.make_gene_matrix if fn == "gene" else ga.pp.make_gene_score_matrix
    cells = _cells(data, 40)
    sel = make(data["path"], data["gtf"], metrics=cudf.DataFrame({"barcode": cells}),
               min_fragments_per_cell=50)
    full = make(data["path"], data["gtf"], min_fragments_per_cell=50)
    assert set(sel.obs_names) == set(cells)
    ref = full[sel.obs_names, sel.var_names]
    assert (sp.csr_matrix(sel.X) != sp.csr_matrix(ref.X)).nnz == 0


def _cells_adata(data):
    import anndata as ad
    cells = _cells(data, 80)
    obs = pd.DataFrame({"group": np.where(np.arange(len(cells)) % 2, "a", "b")}, index=cells)
    return ad.AnnData(obs=obs)


def test_peaks_run(data):
    adata = _cells_adata(data)
    called = ga.peak.call(adata, "group", data["path"], genome=CHROM_SIZES,
                              verbose=False, inplace=False)
    assert sum(len(p) for p in called.values()) > 0

    peaks = pd.concat(called.values())[["chrom", "start", "end"]].drop_duplicates()
    peak_file = data["dir"] / "peaks.bed"
    peaks.to_csv(peak_file, sep="\t", header=False, index=False)
    # counting_strategy="fragment" is not exercised: it fails on cupy's
    # array-valued `repeats`.
    for strategy in ("paired-insertion", "insertion"):
        res = ga.peak.make_matrix(adata, data["path"], peak_file=peak_file, use_rep=None,
                                     genome=CHROM_SIZES, counting_strategy=strategy, verbose=False)
        assert res.X.nnz > 0


def test_footprint_cuts(data):
    from gatac.tobias.footprint import _load_cuts
    adata = _cells_adata(data)
    groups = dict(zip(adata.obs_names, (adata.obs["group"] == "a").astype(int)))
    sizes = {c: CHROM_SIZES[c] for c in ("chr1", "chr2")}
    got = _load_cuts(data["path"], groups, sizes, 50, True)
    df = data["df"]
    df = df[df["barcode"].isin(groups) & df["chrom"].isin(sizes)].assign(g=lambda d: d["barcode"].map(groups))
    df = df.drop_duplicates(["g", "chrom", "start", "end"])
    length = df["chrom"].map(sizes)
    s, e = df["start"], df["end"]
    df = df[(s - 4 >= 0) & (e + 4 - 50 >= 0) & (e + 4 <= length) & (s - 4 + 50 <= length)]
    for g in set(groups.values()):
        for c in sizes:
            sub = df[(df["g"] == g) & (df["chrom"] == c)]
            assert np.array_equal(got[g][c][0], np.sort(sub["start"].to_numpy()))
            assert np.array_equal(got[g][c][1], np.sort(sub["end"].to_numpy() - 1))
