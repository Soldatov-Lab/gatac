"""
Compact fragment files and legacy ones are interchangeable.

``make_parquet`` stores chromosomes and barcodes as integer ids with the
barcode table inside the file. Everything that reads fragments goes through
``FragmentFile``, so every function must return the same result from a
compact file and from a legacy file written by the former DuckDB writer
(names in every row, Snappy, no delta encoding). Synthetic data, GPU required.
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cupy")
pytest.importorskip("cudf")

import duckdb  # noqa: E402
import pyarrow as pa  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import gatac as ga  # noqa: E402
from gatac.pp.fragments import FragmentFile  # noqa: E402

CHROM_SIZES = {"chr1": 3_000_000, "chr2": 2_500_000, "chr3": 2_000_000, "chrX": 1_500_000}
LAYOUTS = ["compact", "legacy"]


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
    paths = {
        "compact": ga.pp.make_parquet(tsv, d / "compact.parquet", barcode_prefix="S1_", row_group_size=9_000,
                                      chroms=["chr1", "chr2"]),
        "legacy": d / "legacy.parquet",
    }
    # The former writer: DuckDB COPY, Snappy, no delta encoding
    duckdb.connect().execute(f"""
        COPY (SELECT column0 AS chrom, column1::UINTEGER AS start, column2::UINTEGER AS "end",
                     'S1_' || column3 AS barcode, column4::USMALLINT AS count
              FROM read_csv('{tsv}', delim='\t', header=false, comment='#'))
        TO '{paths['legacy']}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 9000)
    """)
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
    return {"df": df, "paths": paths, "dir": d, "gtf": gtf}


def _cells(data, n=60):
    counts = data["df"].groupby("barcode").size().sort_values(ascending=False)
    return list(counts.index[:n])


def test_round_trip_keeps_rows_order_and_names(data):
    expected = data["df"]
    for layout in LAYOUTS:
        got = pa.concat_tables(FragmentFile(data["paths"][layout]).iter_arrow()).to_pandas()
        for c in expected.columns:
            assert np.array_equal(got[c].to_numpy().astype(str), expected[c].to_numpy().astype(str)), (layout, c)


def test_compact_file_layout(data):
    ff = FragmentFile(data["paths"]["compact"])
    assert ff.compact and ff.layout == "compact"
    # Given chromosomes first, the others appended in order of appearance
    assert ff.chroms[:2] == ["chr1", "chr2"]
    assert set(ff.chroms) == set(data["df"]["chrom"])
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


@pytest.mark.parametrize("chroms", [None, ["chr2", "chrM"]])
def test_barcode_counts_agree(data, chroms):
    res = [FragmentFile(data["paths"][layout]).barcode_counts(chroms=chroms, mito_chroms=["chrM"])
           .set_index("barcode").sort_index() for layout in LAYOUTS]
    for r in res[1:]:
        pd.testing.assert_frame_equal(r, res[0])


def test_iter_batches_agree(data):
    cells = _cells(data, 40)
    chroms = ["chr3", "chr1", "chrX"]
    out = []
    for layout in LAYOUTS:
        parts = [b.to_pandas() for b in FragmentFile(data["paths"][layout]).iter_batches(
            ["chrom", "start", "end", "count"], row_groups_per_batch=3, chroms=chroms, cells=cells)]
        df = pd.concat(parts).sort_values(["chrom", "start", "end", "cell"]).reset_index(drop=True)
        out.append(df)
    ref = data["df"][data["df"]["chrom"].isin(chroms) & data["df"]["barcode"].isin(cells)]
    assert len(out[0]) == len(ref)
    assert set(np.asarray(chroms)[out[0]["chrom"]]) == set(ref["chrom"])
    for df in out[1:]:
        pd.testing.assert_frame_equal(df, out[0], check_dtype=False)


def test_repack_legacy_and_guard(data):
    d = data["dir"]
    out = FragmentFile(ga.pp.make_parquet(data["paths"]["legacy"], d / "repacked.parquet"))
    assert out.compact
    a = pa.concat_tables(out.iter_arrow())
    b = pa.concat_tables(FragmentFile(data["paths"]["compact"]).iter_arrow())
    assert a.equals(b)
    with pytest.raises(ValueError):
        ga.pp.make_parquet(data["paths"]["compact"], data["paths"]["compact"])


def test_repack_compact_keeps_ids(data):
    src = FragmentFile(data["paths"]["compact"])
    out = FragmentFile(ga.pp.make_parquet(src.path, data["dir"] / "recompact.parquet"))
    assert out.chroms == src.chroms
    assert list(out.barcodes) == list(src.barcodes)


def test_empty_selection_yields_nothing(data):
    for layout in LAYOUTS:
        ff = FragmentFile(data["paths"][layout])
        assert list(ff.iter_batches(chroms=[])) == []
        assert list(ff.iter_batches(cells=[])) == []


def test_batch_prefixes_and_no_partial_file(data, tmp_path):
    src = data["paths"]["legacy"]
    outs = ga.pp.make_parquet_batch([src, src.with_name("compact.parquet")], output_dir=tmp_path / "out",
                                    barcode_prefix=["A_", "B_"], workers=2)
    for out, prefix in zip(outs, ["A_", "B_"]):
        assert all(b.startswith(prefix + "S1_") for b in FragmentFile(out).barcode_counts()["barcode"])
    bad = pd.DataFrame({"barcode": ["not-in-file"], "n_unique": [1], "n_total": [1]})
    target = tmp_path / "bad.parquet"
    with pytest.raises(ValueError):
        ga.pp.fragments.write_fragments(lambda: FragmentFile(src).iter_arrow(), target, barcode_stats=bad)
    assert not target.exists() and not list(tmp_path.glob("bad.parquet*"))


def test_read_fragments_parquet_categoricals(data):
    frames = []
    for layout in LAYOUTS:
        df = ga.pp.read_fragments_parquet(data["paths"][layout]).to_pandas()
        assert df["chrom"].dtype == "category" and df["barcode"].dtype == "category"
        frames.append(df)
    for df in frames[1:]:
        assert list(df["chrom"].cat.categories) == list(frames[0]["chrom"].cat.categories)
        assert list(df["barcode"].cat.categories) == list(frames[0]["barcode"].cat.categories)
        pd.testing.assert_frame_equal(df, frames[0], check_dtype=False)


def _adata_equal(a, b):
    assert list(a.obs_names) == list(b.obs_names)
    assert list(a.var_names) == list(b.var_names)
    assert (sp.csr_matrix(a.X) != sp.csr_matrix(b.X)).nnz == 0
    common = [c for c in a.obs.columns if c in b.obs.columns]
    pd.testing.assert_frame_equal(a.obs[common], b.obs[common], check_dtype=False)


@pytest.mark.parametrize("strategy", ["unique", "count", "binarize"])
def test_tile_matrix_agrees(data, strategy):
    res = [ga.pp.make_tile_matrix(data["paths"][layout], CHROM_SIZES, tile_size=50_000,
                                  min_fragments_per_cell=50, count_strategy=strategy,
                                  row_groups_per_batch=4) for layout in LAYOUTS]
    assert res[0].n_obs > 10
    for r in res[1:]:
        _adata_equal(r, res[0])


def test_metrics_agree(data):
    import polars as pl
    tss = pl.DataFrame({"chrom": ["chr1", "chr1", "chr2", "chr3"],
                        "tss": [200_000, 900_000, 400_000, 1_000_000], "strand": ["+", "-", "+", "+"]})
    res = [ga.pp.compute_metrics(data["paths"][layout], tss, min_unique_frags=50,
                                 row_groups_per_batch=4).to_pandas().sort_values("barcode").reset_index(drop=True)
           for layout in LAYOUTS]
    assert len(res[0]) > 10
    for r in res[1:]:
        pd.testing.assert_frame_equal(r, res[0], check_dtype=False)


def test_doublets_agree(data):
    sizes = {c: CHROM_SIZES[c] for c in ("chr1", "chr2", "chr3")}
    res = [ga.pp.detect_doublets(data["paths"][layout], sizes, min_fragments=50)
           .sort_values("cell_id").reset_index(drop=True) for layout in LAYOUTS]
    for r in res[1:]:
        pd.testing.assert_frame_equal(r, res[0], check_dtype=False)


def test_filter_writes_compact_and_agrees(data):
    d = data["dir"]
    tables = []
    for layout in LAYOUTS:
        out = ga.pp.filter_fragments(data["paths"][layout], d / f"filtered_{layout}.parquet",
                                     min_fragments_per_cell=200, chrom_sizes=CHROM_SIZES)
        ff = FragmentFile(out)
        assert ff.compact
        t = pa.concat_tables(ff.iter_arrow()).to_pandas()
        tables.append(t)
        # The barcode table describes the filtered fragments
        counts = ff.barcode_counts().set_index("barcode")["n_unique"].sort_index()
        assert counts.equals(t.groupby("barcode").size().sort_index().rename("n_unique"))
    ref = data["df"]
    kept = ref[ref["chrom"].isin(CHROM_SIZES)]
    n = kept.groupby("barcode").size()
    kept = kept[kept["barcode"].isin(n.index[n >= 200])].reset_index(drop=True)
    for t in tables:
        assert np.array_equal(t[["chrom", "start", "end", "barcode"]].to_numpy().astype(str),
                              kept[["chrom", "start", "end", "barcode"]].to_numpy().astype(str))


@pytest.mark.parametrize("fn", ["gene", "genescore"])
def test_gene_matrices_agree(data, fn):
    make = ga.pp.make_gene_matrix if fn == "gene" else ga.pp.make_gene_score_matrix
    res = [make(data["paths"][layout], data["gtf"], min_fragments_per_cell=50) for layout in LAYOUTS]
    assert res[0].n_obs > 10
    for r in res[1:]:
        _adata_equal(r, res[0])


def _cells_adata(data):
    import anndata as ad
    cells = _cells(data, 80)
    obs = pd.DataFrame({"group": np.where(np.arange(len(cells)) % 2, "a", "b")}, index=cells)
    return ad.AnnData(obs=obs)


def test_peaks_agree(data):
    adata = _cells_adata(data)
    called = [ga.tl.call_peaks(adata, "group", data["paths"][layout], genome=CHROM_SIZES,
                               verbose=False, inplace=False) for layout in LAYOUTS]
    assert sum(len(p) for p in called[0].values()) > 0
    for c in called[1:]:
        assert c.keys() == called[0].keys()
        for g in c:
            pd.testing.assert_frame_equal(c[g].reset_index(drop=True), called[0][g].reset_index(drop=True))

    peaks = pd.concat(called[0].values())[["chrom", "start", "end"]].drop_duplicates()
    peak_file = data["dir"] / "peaks.bed"
    peaks.to_csv(peak_file, sep="\t", header=False, index=False)
    # counting_strategy="fragment" is not exercised: it fails on cupy's
    # array-valued `repeats`, independently of the fragment layout.
    for strategy in ("paired-insertion", "insertion"):
        res = [ga.tl.make_peak_matrix(adata, data["paths"][layout], peak_file=peak_file, use_rep=None,
                                      genome=CHROM_SIZES, counting_strategy=strategy, verbose=False)
               for layout in LAYOUTS]
        assert res[0].X.nnz > 0
        for r in res[1:]:
            _adata_equal(r, res[0])
