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

import anndata as ad  # noqa: E402
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
    meta = pq.ParquetFile(data["path"]).metadata.metadata
    assert {b"gatac.format_version", b"gatac.chroms", b"gatac.n_barcodes"} <= set(meta)
    # The prefix is in the footer only; the barcode table keeps the TSV's names
    assert meta[b"gatac.barcode_prefix"] == b"S1_" and ff.barcode_prefix == "S1_"
    # cuDF reads the barcode table too (it reads DELTA_BYTE_ARRAY strings
    # written with a page index back as NUL bytes)
    import cudf
    names = cudf.read_parquet(data["path"], columns=["bc_name"])["bc_name"].dropna()
    assert ["S1_" + n for n in names.to_arrow().to_pylist()] == list(ff.barcodes)


def test_stored_columns_and_encodings(data):
    """Lengths, not ends; barcode ids, lengths and counts byte-stream split."""
    pf = pq.ParquetFile(data["path"])
    schema = pf.schema_arrow
    assert schema.names[:5] == ["chrom_id", "start", "length", "barcode_id", "count"]
    assert schema.field("length").type == pa.uint32() and schema.field("count").type == pa.uint32()
    rg = pf.metadata.row_group(0)
    enc = {rg.column(i).path_in_schema: set(rg.column(i).encodings) for i in range(rg.num_columns)}
    for c in ("length", "barcode_id", "count"):
        assert "BYTE_STREAM_SPLIT" in enc[c], (c, enc[c])
    assert "DELTA_BINARY_PACKED" in enc["start"]


def test_end_before_start_is_rejected(tmp_path):
    t = pa.table({"chrom": ["chr1", "chr1"], "start": pa.array([100, 500], pa.uint32()),
                  "end": pa.array([200, 400], pa.uint32()), "barcode": ["A", "A"],
                  "count": pa.array([1, 1], pa.uint16())})
    with pytest.raises(ValueError, match="ends before it starts"):
        ga.io.fragments.write_fragments(lambda: iter([t]), tmp_path / "bad.parquet")


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


def test_batch_same_file_names(data, tmp_path, caplog):
    # Cell Ranger names every sample's file atac_fragments.tsv.gz; without
    # barcode prefixes their cells share names, which is reported
    inputs = []
    for sample in ("A", "B"):
        (tmp_path / sample).mkdir()
        inputs.append(tmp_path / sample / "atac_fragments.tsv.gz")
        shutil.copy(data["tsv"], inputs[-1])
    with pytest.raises(ValueError, match="file_prefix"):
        ga.io.make_parquet_batch(inputs, output_dir=tmp_path / "out", workers=2)
    assert not (tmp_path / "out").exists()
    outs = ga.io.make_parquet_batch(inputs, output_dir=tmp_path / "out", workers=2,
                                    file_prefix=["A_", "B_"])
    assert [p.name for p in outs] == ["A_atac_fragments.parquet", "B_atac_fragments.parquet"]
    assert all(p.exists() for p in outs)
    assert "share 300 barcode names" in caplog.text


def test_sample_names():
    names = ga.io.sample_names
    assert names(["samples/s1.tsv.gz", "samples/s2.tsv.gz"]) == ["s1", "s2"]
    assert names(["runs/A/outs/atac_fragments.tsv.gz", "runs/B/outs/atac_fragments.tsv.gz"]) == ["A", "B"]
    assert names(["r1/A/f.tsv.gz", "r1/B/f.tsv.gz", "r2/A/f.tsv.gz"]) == ["r1_A", "r1_B", "r2_A"]
    assert names(["data/pbmc.bed.gz"]) == ["pbmc"]


def test_convert_cli_sample_placeholders(data, tmp_path, monkeypatch):
    """Cell Ranger outputs: {sample} names the output files and the cells after the run directory."""
    from gatac import cli

    for s in ("A", "B"):
        (tmp_path / "runs" / s / "outs").mkdir(parents=True)
        shutil.copy(data["tsv"], tmp_path / "runs" / s / "outs" / "atac_fragments.tsv.gz")
    out = tmp_path / "parquets"

    def convert(*options):
        monkeypatch.setattr("sys.argv", ["gatac", "convert", str(tmp_path / "runs/*/outs/atac_fragments.tsv.gz"),
                                         "--output-dir", str(out), "-j", "2", *options])
        cli.main()

    with pytest.raises(SystemExit):  # both would be written to parquets/atac_fragments.parquet
        convert()
    convert("--file-prefix", "{sample}_", "--barcode-prefix", "{sample}#")
    paths = [out / f"{s}_atac_fragments.parquet" for s in ("A", "B")]
    assert [FragmentFile(p).barcode_prefix for p in paths] == ["A#", "B#"]
    assert ga.pp.scan_tiles(paths, **SCAN_KW).obs_names.is_unique


def test_convert_names_per_file(data, tmp_path, monkeypatch):
    """Per-file outputs and prefixes, as lists, a {sample} output or a sample sheet, from the CLI and Python."""
    from gatac import cli

    inputs = []
    for s in ("s1", "s2"):
        inputs.append(tmp_path / "samples" / f"{s}.tsv.gz")
        inputs[-1].parent.mkdir(exist_ok=True)
        shutil.copy(data["tsv"], inputs[-1])

    def convert(*argv):
        monkeypatch.setattr("sys.argv", ["gatac", "convert", *map(str, argv)])
        cli.main()

    def prefixes(paths):
        return [FragmentFile(p).barcode_prefix for p in paths]

    # Lists, in input order
    outs = [tmp_path / "lists" / "patient1.parquet", tmp_path / "lists" / "patient2.parquet"]
    convert(*inputs, "-o", *outs, "--barcode-prefix", "patient1_", "patient2_")
    assert prefixes(outs) == ["patient1_", "patient2_"]
    with pytest.raises(SystemExit):  # 2 inputs, 3 prefixes
        convert(*inputs, "-o", *outs, "--barcode-prefix", "a_", "b_", "c_")

    # One output with {sample}
    convert(*inputs, "-o", tmp_path / "tpl" / "{sample}.parquet", "--barcode-prefix", "{sample}_")
    assert prefixes([tmp_path / "tpl" / f"{s}.parquet" for s in ("s1", "s2")]) == ["s1_", "s2_"]

    # A sample sheet; an empty cell takes the default
    sheet = tmp_path / "samples.csv"
    sheet.write_text("fragments,output,barcode_prefix\n"
                     f"{inputs[0]},{tmp_path / 'sheet' / 'p1.parquet'},P1_\n"
                     f"{inputs[1]},{tmp_path / 'sheet' / 'p2.parquet'},\n")
    convert("--samples", sheet)
    assert prefixes([tmp_path / "sheet" / "p1.parquet", tmp_path / "sheet" / "p2.parquet"]) == ["P1_", ""]
    got = ga.io.make_parquet_batch(samples=pd.DataFrame({"fragments": inputs, "barcode_prefix": ["X_", "Y_"]}),
                                   output_dir=tmp_path / "frame")
    assert [p.name for p in got] == ["s1.parquet", "s2.parquet"] and prefixes(got) == ["X_", "Y_"]
    with pytest.raises(ValueError, match="sample sheet"):
        ga.io.make_parquet_batch(samples=pd.DataFrame({"file": inputs}))


def test_to_tsv_round_trip(data, tmp_path):
    """The TSV is the original one: barcodes without the file's prefix."""
    cols = list(data["df"].columns)
    gz = ga.io.to_tsv(data["path"], tmp_path / "frags.tsv.gz")
    plain = FragmentFile(data["path"]).to_tsv(tmp_path / "frags.tsv")
    original = pd.read_csv(data["tsv"], sep="\t", header=None, names=cols, comment="#")
    for out in (gz, plain):
        got = pd.read_csv(out, sep="\t", header=None, names=cols)
        pd.testing.assert_frame_equal(got, original, check_dtype=False)
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



def test_filter_writes_the_same_layout(data):
    """The GPU writer of filter_fragments writes what make_parquet writes."""
    out = ga.pp.filter_fragments(data["path"], data["dir"] / "filtered_layout.parquet",
                                 min_fragments_per_cell=200, chrom_sizes=CHROM_SIZES)
    ref, got = pq.ParquetFile(data["path"]), pq.ParquetFile(out)
    assert got.schema_arrow == ref.schema_arrow

    # cuDF names dictionary pages the v1 way, and leaves a column chunk
    # uncompressed when ZSTD would not make it smaller
    def encodings(col):
        names = {"PLAIN_DICTIONARY": "RLE_DICTIONARY"}
        return {names.get(e, e) for e in col.encodings} - {"PLAIN", "RLE"}

    for i in range(ref.metadata.num_columns):
        a, b = ref.metadata.row_group(0).column(i), got.metadata.row_group(0).column(i)
        assert encodings(b) == encodings(a), a.path_in_schema
        assert b.compression in ("ZSTD", "UNCOMPRESSED"), a.path_in_schema
        assert b.has_column_index and b.has_offset_index, a.path_in_schema
    ff = FragmentFile(out)
    expected = pa.concat_tables(ff.iter_arrow()).to_pandas()
    # DuckDB and cuDF's chunked reader (data page v1 only) read it
    with duckdb.connect() as con:
        ff.register(con)
        assert con.execute("SELECT count(*) FROM fragments").fetchone()[0] == len(expected)
    got = ff.read_categorical(["chrom", "start", "end", "barcode", "count"], low_memory=True)
    assert np.array_equal(got["start"].to_numpy(), expected["start"].to_numpy())


def test_filter_several_files_prefix_and_no_cell(data):
    """Files filtered at once, each given its prefix; a file with no kept cell gives an empty file."""
    other = data["dir"] / "other.parquet"
    shutil.copy(data["path"], other)
    cells = _cells(data, 20)
    # Cells are matched on the input's names, then take the new prefix
    metrics = pd.DataFrame({"barcode": cells})
    outs = [data["dir"] / "f1.parquet", data["dir"] / "f2.parquet"]
    ga.pp.filter_fragments([data["path"], other], outs, metrics=metrics,
                           barcode_prefix=["X_", "Y_"], chrom_sizes=CHROM_SIZES)
    a = pa.concat_tables(FragmentFile(outs[0]).iter_arrow()).to_pandas()
    assert set(a["barcode"]) == {"X_" + c.removeprefix("S1_") for c in cells}
    ref = data["df"]
    assert len(a) == (ref["chrom"].isin(CHROM_SIZES) & ref["barcode"].isin(cells)).sum()
    b = pa.concat_tables(FragmentFile(outs[1]).iter_arrow()).to_pandas()
    pd.testing.assert_frame_equal(b, a.assign(barcode="Y_" + a["barcode"].str[2:]))
    # Without barcode_prefix the output keeps the input's
    assert FragmentFile(ga.pp.filter_fragments(
        outs[1], data["dir"] / "f4.parquet", min_fragments_per_cell=1)).barcode_prefix == "Y_"
    # No cell of the metrics in this file
    out = ga.pp.filter_fragments(data["path"], data["dir"] / "f3.parquet",
                                 metrics=pd.DataFrame({"barcode": ["nope"]}))
    ff = FragmentFile(out)
    assert ff.num_rows == 0 and ff.n_barcodes == 0


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


@pytest.fixture
def two_samples(data, tmp_path):
    """Two fragment files of the same barcodes (one's are a subset of the other's), with their own prefixes."""
    small = ga.pp.filter_fragments(data["path"], tmp_path / "small.parquet", metrics=pd.DataFrame(
        {"barcode": _cells(data, 20)}), chrom_sizes=CHROM_SIZES, barcode_prefix="A_")
    return [small, ga.io.set_barcode_prefix(data["path"], "B_", output_path=tmp_path / "big.parquet")]


def test_set_barcode_prefix(data, tmp_path):
    """Only the footer changes; every reader sees the new names, to_tsv the original ones."""
    copy = tmp_path / "copy.parquet"
    shutil.copy(data["path"], copy)
    before = copy.read_bytes()
    ga.io.set_barcode_prefix(copy, "new#")
    after = copy.read_bytes()
    n = len(before) - 8 - int.from_bytes(before[-8:-4], "little")
    assert after[:n] == before[:n]
    ff = FragmentFile(copy)
    assert ff.barcode_prefix == "new#"
    assert list(ff.barcodes) == ["new#" + b.removeprefix("S1_") for b in FragmentFile(data["path"]).barcodes]
    # pyarrow keeps the other footer keys, cuDF and DuckDB read the data
    meta = pq.ParquetFile(copy).metadata.metadata
    assert {k: v for k, v in meta.items() if k != b"gatac.barcode_prefix"} == {
        k: v for k, v in pq.ParquetFile(data["path"]).metadata.metadata.items() if k != b"gatac.barcode_prefix"}
    assert pa.concat_tables(ff.iter_arrow()).drop(["barcode"]).equals(
        pa.concat_tables(FragmentFile(data["path"]).iter_arrow()).drop(["barcode"]))
    assert ff.read_categorical(["start"])["start"].to_numpy().tolist() == data["df"]["start"].tolist()
    with duckdb.connect() as con:
        ff.register(con)
        assert con.execute("SELECT count(*) FROM fragments WHERE barcode LIKE 'new#%'").fetchone()[0] == len(data["df"])
    tsv = [ga.io.to_tsv(p, tmp_path / f"{i}.tsv").read_bytes() for i, p in enumerate((copy, data["path"]))]
    assert tsv[0] == tsv[1]
    # A renamed copy leaves the input as it was; "" removes the prefix
    bare = ga.io.set_barcode_prefix(copy, "", output_path=tmp_path / "bare.parquet")
    assert FragmentFile(copy).barcode_prefix == "new#"
    assert list(FragmentFile(bare).barcodes) == [b.removeprefix("new#") for b in ff.barcodes]
    assert not list(tmp_path.glob("*.tmp"))


def test_footer_key_values_round_trip(tmp_path):
    """Lists of 15 or more keys take the long list header."""
    from gatac.io._footer import update_key_value_metadata

    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3]}), path)
    many = {f"k{i}": "v" * i for i in range(20)}
    update_key_value_metadata(path, many)
    meta = pq.ParquetFile(path).metadata.metadata
    assert {k.decode(): v.decode() for k, v in meta.items() if k.startswith(b"k")} == many
    assert b"ARROW:schema" in meta and pq.read_table(path)["x"].to_pylist() == [1, 2, 3]


def test_shared_cell_names_are_refused(data, tmp_path):
    """Combining files whose cells share names raises, before any work, with the fix."""
    other = tmp_path / "other.parquet"
    shutil.copy(data["path"], other)
    for make in (lambda p: ga.pp.scan_tiles(p, **SCAN_KW),
                 lambda p: ga.pp.make_gene_matrix(p, data["gtf"]),
                 lambda p: ga.pp.make_gene_score_matrix(p, data["gtf"])):
        with pytest.raises(ValueError, match="gatac rename"):
            make([data["path"], other])
    ga.io.set_barcode_prefix(other, "S2_")
    assert ga.pp.scan_tiles([data["path"], other], **SCAN_KW).obs_names.is_unique



SCAN_KW = dict(chrom_sizes=CHROM_SIZES, tile_size=50_000, min_fragments_per_cell=50)


@pytest.mark.parametrize("strategy", ["unique", "count", "binarize"])
def test_scan_tiles_matches_the_tile_matrices(two_samples, strategy):
    """The scan's cells are the per-file matrices' and its statistics are their column sums."""
    one = [ga.pp.make_tile_matrix(p, count_strategy=strategy, **SCAN_KW) for p in two_samples]
    tiles = ga.pp.scan_tiles(two_samples, count_strategy=strategy, row_groups_per_batch=1, **SCAN_KW)
    assert tiles.X is None
    X = sp.vstack([o.X for o in one]).tocsc().astype(np.int64)
    np.testing.assert_array_equal(tiles.var["n_cells"], np.diff(X.indptr))
    np.testing.assert_array_equal(tiles.var["count"], np.asarray(X.sum(axis=0)).ravel())
    np.testing.assert_array_equal(tiles.var["count_sq"], np.asarray(X.multiply(X).sum(axis=0)).ravel())
    assert list(tiles.var_names) == list(one[0].var_names)
    names = [n for o in one for n in o.obs_names]
    assert list(tiles.obs_names) == names and list(tiles.obs["barcode"]) == names
    np.testing.assert_array_equal(tiles.obs["n_unique"], np.concatenate([o.obs["n_unique"] for o in one]))
    assert list(tiles.obs["source_file"]) == [p.name for p, o in zip(two_samples, one) for _ in range(o.n_obs)]


def test_build_tile_matrix(two_samples, tmp_path):
    """In memory or written, all tiles or the selected ones, every cell or some."""
    one = [ga.pp.make_tile_matrix(p, **SCAN_KW) for p in two_samples]
    full = sp.vstack([o.X for o in one]).tocsr()
    tiles = ga.pp.scan_tiles(two_samples, **SCAN_KW)

    built = ga.pp.build_tile_matrix(tiles, features=None)
    assert (built.X != full).nnz == 0 and built.X.has_sorted_indices
    assert list(built.obs_names) == list(tiles.obs_names)

    ga.pp.select_features(tiles, n_features=40)
    alone = ad.AnnData(full, obs=tiles.obs, var=tiles.var[["chrom", "start", "end"]].copy())
    ga.pp.select_features(alone, n_features=40)
    np.testing.assert_array_equal(tiles.var["selected"], alone.var["selected"])

    sel = tiles.var["selected"].to_numpy()
    built = ga.pp.build_tile_matrix(tiles)
    assert (built.X != full[:, sel]).nnz == 0
    written = ga.pp.build_tile_matrix(tiles, output_path=tmp_path / "c.h5ad")
    assert written.isbacked
    assert (written.X.to_memory() != built.X).nnz == 0
    assert list(written.var_names) == list(built.var_names)
    written.file.close()

    # cells dropped before the build: the written matrix is cut to size
    keep = np.arange(tiles.n_obs) % 3 != 0
    some = tiles[keep].copy()
    ga.pp.build_tile_matrix(some, output_path=tmp_path / "d.h5ad").file.close()
    assert (ad.read_h5ad(tmp_path / "d.h5ad").X != full[keep][:, sel]).nnz == 0

    # make_tile_matrix on a list runs the three steps
    assert (ga.pp.make_tile_matrix(two_samples, n_features=40, **SCAN_KW).X != built.X).nnz == 0
    every = ga.pp.make_tile_matrix(two_samples, **SCAN_KW)
    assert (every.X != full).nnz == 0 and list(every.obs_names) == list(tiles.obs_names)
    alone = ga.pp.make_tile_matrix(two_samples[1], n_features=40, **SCAN_KW)
    assert alone.n_vars == 40 and alone.n_obs == one[1].n_obs
    with pytest.raises(ValueError, match="together"):
        ga.pp.build_tile_matrix(tiles[np.r_[0, tiles.n_obs - 1, 1]].copy())


def test_metrics_independent_of_batching(data):
    tss = pd.DataFrame({"chrom": ["chr1", "chr1", "chr2", "chr3"],
                        "tss": [200_000, 900_000, 400_000, 1_000_000], "strand": ["+", "-", "+", "+"]})
    res = [ga.pp.compute_metrics(data["path"], tss, min_unique_frags=50,
                                 row_groups_per_batch=rg).to_pandas().sort_values("barcode").reset_index(drop=True)
           for rg in (1, 4, 64)]
    assert len(res[0]) > 10
    for r in res[1:]:
        pd.testing.assert_frame_equal(r, res[0], check_dtype=False)


def test_metrics_several_files(data, tmp_path):
    """A list of files gives each file's rows, in the order given."""
    tss = pd.DataFrame({"chrom": ["chr1", "chr2"], "tss": [200_000, 400_000], "strand": ["+", "-"]})
    small = ga.pp.filter_fragments(data["path"], tmp_path / "small.parquet", metrics=pd.DataFrame(
        {"barcode": _cells(data, 20)}), chrom_sizes=CHROM_SIZES)
    one = [ga.pp.compute_metrics(p, tss, min_unique_frags=50).to_pandas() for p in (small, data["path"])]
    both = ga.pp.compute_metrics([small, data["path"]], tss, min_unique_frags=50, workers=2).to_pandas()
    pd.testing.assert_frame_equal(both, pd.concat(one, ignore_index=True))


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


def test_gene_matrix_var_spans_transcripts(data, tmp_path):
    # Genes with up to three transcripts of different extents: each gene's
    # coordinates are the span of its transcripts' domains, the same every run
    rng = np.random.default_rng(2)
    rows = []
    for i in range(40):
        c = list(CHROM_SIZES)[i % 4]
        strand = "+" if i % 2 else "-"
        gene_attrs = f'gene_id "G{i}"; gene_name "GENE{i}";'
        s0 = int(rng.integers(5_000, CHROM_SIZES[c] - 80_000))
        for t in range(1 + i % 3):
            s = s0 + int(rng.integers(0, 10_000))
            e = s + int(rng.integers(2_000, 50_000))
            if t == 0:
                rows.append(f"{c}\ttest\tgene\t{s0}\t{s0 + 70_000}\t.\t{strand}\t.\t{gene_attrs}")
            attrs = f'{gene_attrs} transcript_id "T{i}.{t}"; transcript_name "GENE{i}-20{t}";'
            rows.append(f"{c}\ttest\ttranscript\t{s}\t{e}\t.\t{strand}\t.\t{attrs}")
    gtf = tmp_path / "multi.gtf"
    gtf.write_text("\n".join(rows) + "\n")

    runs = [ga.pp.make_gene_matrix(data["path"], gtf, output_path=tmp_path / f"genes{k}.h5ad",
                                   min_fragments_per_cell=50).var for k in range(2)]
    pd.testing.assert_frame_equal(runs[0], runs[1])
    domains = ga.pp.gene.load_gene_annotation(gtf)
    exp = domains.groupby("id").agg(start=("start", "min"), end=("end", "max"))
    got = runs[0].set_index("id")[["start", "end"]]
    pd.testing.assert_frame_equal(got.sort_index(), exp.loc[got.index].sort_index(), check_dtype=False)
    assert (domains.groupby("id").size() > 1).sum() > 10


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


@pytest.mark.parametrize("fn", ["gene", "genescore"])
def test_gene_matrices_of_several_files(two_samples, data, fn, tmp_path):
    """A list of files makes the per-file matrices' rows, in memory or written as built."""
    make = ga.pp.make_gene_matrix if fn == "gene" else ga.pp.make_gene_score_matrix
    kw = dict(min_fragments_per_cell=50)
    one = [make(p, data["gtf"], output_path=tmp_path / f"one{i}.h5ad", **kw)
           for i, p in enumerate(two_samples)]
    full = sp.vstack([sp.csr_matrix(o.X) for o in one]).tocsr()

    both = make(two_samples, data["gtf"], **kw)
    assert (sp.csr_matrix(both.X) != full).nnz == 0
    assert list(both.var_names) == list(one[0].var_names)
    assert list(both.obs_names) == [n for o in one for n in o.obs_names]
    assert list(both.obs["source_file"]) == [p.name for p, o in zip(two_samples, one) for _ in range(o.n_obs)]

    # written as built
    written = make(two_samples, data["gtf"], output_path=tmp_path / "both.h5ad", **kw)
    assert written.isbacked and list(written.obs_names) == list(both.obs_names)
    X = written.X.to_memory()
    assert X.dtype == both.X.dtype and (X != full).nnz == 0
    assert list(written.var_names) == list(both.var_names)
    written.file.close()


def test_combine_file_matrices_widens_and_checks_features(tmp_path):
    from gatac.pp._combine import combine_file_matrices

    def part(prefix, dtype, top):
        names = [f"{prefix}c1", f"{prefix}c2"]
        return ad.AnnData(sp.csr_matrix(np.array([[0, 1, 0], [top, 0, 2]], dtype=dtype)),
                          obs=pd.DataFrame({"barcode": names}, index=names),
                          var=pd.DataFrame(index=["g1", "g2", "g3"]))

    parts = [part(p, dtype, top)
             for p, dtype, top in (("a_", np.uint8, 200), ("b_", np.uint16, 60_000), ("c_", np.int32, 100_000))]
    expected = sp.vstack([p.X for p in parts]).tocsr()
    for out in (None, tmp_path / "c.h5ad"):
        res = combine_file_matrices(iter(parts), ["a", "b", "c"], out)
        X = res.X.to_memory() if res.isbacked else res.X
        assert X.dtype == np.int32 and (X != expected).nnz == 0
        assert list(res.obs_names) == ["a_c1", "a_c2", "b_c1", "b_c2", "c_c1", "c_c2"]
        assert list(res.obs["source_file"]) == list("aabbcc")
        if res.isbacked:
            res.file.close()
    other = ad.AnnData(sp.csr_matrix((1, 2)), var=pd.DataFrame(index=["g1", "x"]))
    with pytest.raises(ValueError, match="other features"):
        combine_file_matrices([parts[0], other], ["a", "b"])
    with pytest.raises(ValueError, match=r"2 cell names are in more than one of the files \['a', 'c'\]"):
        combine_file_matrices([parts[0], parts[1], part("a_", np.uint8, 1)], ["a", "b", "c"])


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


def test_peaks_from_a_directory_of_samples(data, tmp_path):
    """A tile matrix of several files finds each cell's fragments by its source_file."""
    cells = _cells(data, 80)
    d = tmp_path / "filtered"
    d.mkdir()
    paths = [ga.pp.filter_fragments(data["path"], d / f"{name}.parquet",
                                    metrics=pd.DataFrame({"barcode": part}), chrom_sizes=CHROM_SIZES)
             for name, part in (("s1", cells[::2]), ("s2", cells[1::2]))]
    adata = ga.pp.make_tile_matrix(paths, **SCAN_KW)
    assert set(adata.obs["source_file"]) == {"s1.parquet", "s2.parquet"}
    adata.obs["group"] = np.where(np.arange(adata.n_obs) % 2, "a", "b")

    kw = dict(genome=CHROM_SIZES, verbose=False)
    by_dir = ga.peak.call(adata, "group", d, inplace=False, **kw)
    by_file = ga.peak.call(adata, "group", data["path"], inplace=False, **kw)
    for g in by_file:
        pd.testing.assert_frame_equal(by_dir[g], by_file[g])

    peak_file = tmp_path / "peaks.bed"
    pd.concat(by_file.values())[["chrom", "start", "end"]].drop_duplicates().to_csv(
        peak_file, sep="\t", header=False, index=False)
    kw.update(peak_file=peak_file, use_rep=None)
    X_dir = ga.peak.make_matrix(adata, d, **kw).X
    X_file = ga.peak.make_matrix(adata, data["path"], **kw).X
    assert X_dir.nnz > 0 and (X_dir != X_file).nnz == 0


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
