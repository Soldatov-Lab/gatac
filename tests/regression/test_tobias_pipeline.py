"""footprint -> bindetect end to end on a small synthetic genome."""
import gzip

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cupy")
pytest.importorskip("cudf")

import anndata as ad  # noqa: E402

import gatac as ga  # noqa: E402

CHROMS = {"chr1": 400_000, "chr2": 300_000}
MOTIFS = {                                   # A C G T per position
    "AP1": ["0.9 0.0 0.1 0.0", "0.0 0.0 0.0 1.0", "0.0 0.0 1.0 0.0", "1.0 0.0 0.0 0.0",
            "0.0 1.0 0.0 0.0", "0.0 0.0 0.0 1.0", "1.0 0.0 0.0 0.0"],
    "CTCFLIKE": ["0.0 1.0 0.0 0.0", "0.0 1.0 0.0 0.0", "1.0 0.0 0.0 0.0", "0.0 0.0 1.0 0.0",
                 "0.0 0.0 1.0 0.0", "0.0 0.0 1.0 0.0", "0.0 0.0 1.0 0.0", "0.0 1.0 0.0 0.0"],
    "GATA": ["1.0 0.0 0.0 0.0", "0.0 0.0 1.0 0.0", "1.0 0.0 0.0 0.0", "0.0 0.0 0.0 1.0",
             "1.0 0.0 0.0 0.0", "1.0 0.0 0.0 0.0"],
}


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("tobias")
    rng = np.random.default_rng(0)

    fasta = d / "genome.fa"
    with open(fasta, "w") as fh:
        for c, n in CHROMS.items():
            seq = "".join(rng.choice(list("ACGT"), n))
            fh.write(f">{c}\n" + "\n".join(seq[i:i + 60] for i in range(0, n, 60)) + "\n")

    peaks = []
    for c, n in CHROMS.items():
        for s in np.sort(rng.choice(np.arange(5_000, n - 5_000, 1_500), 60, replace=False)):
            peaks.append((c, int(s), int(s) + 500))
    peaks = pd.DataFrame(peaks, columns=["chrom", "start", "end"])

    # Two groups of cells: fragments in and around the peaks, the rest spread over the genome
    cells = {g: [f"{g}{i:03d}-1" for i in range(40)] for g in ("a", "b")}
    rows = []
    for g, bcs in cells.items():
        for _ in range(60_000):
            if rng.random() < 0.6:
                p = peaks.iloc[rng.integers(len(peaks))]
                c, s = p.chrom, int(rng.integers(p.start - 200, p.end + 200))
            else:
                c = ("chr1", "chr2")[int(rng.random() < 0.4)]
                s = int(rng.integers(1_000, CHROMS[c] - 2_000))
            rows.append((c, s, s + int(rng.integers(40, 400)), bcs[rng.integers(len(bcs))], 1))
    frags = pd.DataFrame(rows, columns=["chrom", "start", "end", "barcode", "count"])
    frags = frags.sort_values(["chrom", "start", "end"]).drop_duplicates(["chrom", "start", "end", "barcode"])
    tsv = d / "frags.tsv.gz"
    with gzip.open(tsv, "wt") as fh:
        frags.to_csv(fh, sep="\t", header=False, index=False)
    parquet = ga.io.make_parquet(tsv, d / "frags.parquet")

    meme = d / "motifs.meme"
    with open(meme, "w") as fh:
        fh.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
                 "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for name, rows_ in MOTIFS.items():
            fh.write(f"MOTIF {name} {name}\nletter-probability matrix: alength= 4 w= {len(rows_)} nsites= 20 E= 0\n")
            fh.write("\n".join(f" {r}" for r in rows_) + "\n\n")

    obs = pd.DataFrame({"group": pd.Categorical([g for g, bcs in cells.items() for _ in bcs])},
                       index=[b for bcs in cells.values() for b in bcs])
    adata = ad.AnnData(obs=obs)
    tracks = ga.tobias.footprint(adata, "group", parquet, fasta, peaks=peaks, verbose=False)
    return dict(tracks=tracks, meme=meme)


@pytest.mark.parametrize("deduplicate", [True, False])
@pytest.mark.parametrize("read_length", [None, 50])
def test_load_cuts_matches_unique_rows(tmp_path, deduplicate, read_length):
    # Reference: the former implementation (unique over all rows, then partition)
    from gatac.tobias.footprint import _load_cuts
    rng = np.random.default_rng(3)
    n = 30_000
    chrom = rng.choice(list(CHROMS), n)
    start = rng.integers(0, 300_000, n)
    start[rng.random(n) < 0.3] = 150_000                      # many shared coordinates
    end = start + rng.integers(20, 600, n)
    end[chrom == "chr2"] = np.minimum(end[chrom == "chr2"], CHROMS["chr2"])
    bcs = np.array([f"c{i:02d}-1" for i in range(20)])
    df = pd.DataFrame({"chrom": chrom, "start": start, "end": end,
                       "barcode": bcs[rng.integers(0, 20, n)], "count": 1})
    df = df.sort_values(["chrom", "start", "end"]).drop_duplicates(["chrom", "start", "end", "barcode"])
    tsv = tmp_path / "f.tsv.gz"
    with gzip.open(tsv, "wt") as fh:
        df.to_csv(fh, sep="\t", header=False, index=False)
    parquet = ga.io.make_parquet(tsv, tmp_path / "f.parquet")
    groups = {b: i % 3 for i, b in enumerate(bcs[:18])}       # two barcodes in no group
    got = _load_cuts(parquet, groups, CHROMS, read_length, deduplicate)

    ref = df[df["barcode"].isin(groups)].assign(g=lambda d: d["barcode"].map(groups))
    ref = ref[["g", "chrom", "start", "end"]]
    if deduplicate:
        ref = ref.drop_duplicates()
    for g in range(3):
        for c, length in CHROMS.items():
            sub = ref[(ref["g"] == g) & (ref["chrom"] == c)]
            s, e = sub["start"].to_numpy().astype(np.int64), sub["end"].to_numpy().astype(np.int64)
            ok = ((s >= 0) & (e <= length) if read_length is None else
                  (s - 4 >= 0) & (e + 4 - read_length >= 0) & (e + 4 <= length) & (s - 4 + read_length <= length))
            fwd, rev = got[g][c]
            assert np.array_equal(fwd, np.sort(s[ok])) and np.array_equal(rev, np.sort(e[ok] - 1))
            assert fwd.dtype == np.int64 and rev.dtype == np.int64


def test_bindetect_site_table_and_log2fc(inputs):
    res = ga.tobias.bindetect(inputs["tracks"], inputs["meme"], verbose=False)
    assert res.comparisons == [("a", "b")]
    assert len(res.results) == len(MOTIFS) and len(res.sites) > 0
    cols = list(res.sites.columns)
    assert {"a_score", "b_score", "a_bound", "b_bound"} <= set(cols)
    assert not [c for c in cols if c.endswith("_log2fc")]

    l2 = res.site_log2fc("a", "b")
    s1, s2 = res.sites["a_score"].to_numpy(), res.sites["b_score"].to_numpy()
    expected = np.round(np.log2((s1 + res.pseudo) / (s2 + res.pseudo)), 5)
    assert l2.name == "a_b_log2fc" and l2.index.equals(res.sites.index)
    assert np.array_equal(l2.to_numpy(), expected)
    assert np.array_equal(res.site_log2fc("b", "a").to_numpy(), np.round(np.log2((s2 + res.pseudo) / (s1 + res.pseudo)), 5))
    with pytest.raises(ValueError):
        res.site_log2fc("a", "c")


def test_bindetect_results_do_not_depend_on_return_sites(inputs):
    sites = ga.tobias.bindetect_sites(inputs["tracks"], inputs["meme"], verbose=False)
    full = ga.tobias.bindetect(inputs["tracks"], sites=sites, verbose=False)
    lean = ga.tobias.bindetect(inputs["tracks"], sites=sites, return_sites=False, verbose=False)
    assert lean.sites is None
    pd.testing.assert_frame_equal(full.results, lean.results, check_exact=True)
    with pytest.raises(ValueError):
        lean.site_log2fc("a", "b")
