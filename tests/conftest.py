"""
One small synthetic dataset shared by the tests in ``tests/api``.

Three groups of cells, each enriched in its own set of hotspots. Hotspot
centres of group ``g`` carry a planted copy of motif ``M{g}``; motif
``DECOY`` is planted nowhere. Tests can therefore check results against a
known truth: which groups separate, where peaks are, which motif belongs to
which group.

Session-scoped fixtures build each object once; the function-scoped
fixtures hand every test its own copy, so in-place functions cannot leak
state between tests.
"""
from __future__ import annotations

import gzip

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import gatac as ga

CHROM_SIZES = {"chr1": 2_000_000, "chr2": 1_500_000, "chrX": 1_000_000}
N_GROUPS, CELLS_PER_GROUP, HOTSPOTS_PER_CHROM = 3, 80, 40
MOTIFS = {"M0": "TGACTCATCAG", "M1": "AGATAAGGCTT", "M2": "CCGCGTGGCAA", "DECOY": "GTTACCTAGCA"}


def _hotspots(rng) -> pd.DataFrame:
    """Non-overlapping hotspot centres, assigned to groups round-robin."""
    rows = []
    for c, n in CHROM_SIZES.items():
        k = HOTSPOTS_PER_CHROM * N_GROUPS
        centres = np.sort(rng.choice(np.arange(20_000, n - 20_000, 5_000), k, replace=False))
        rows += [(c, int(s), g) for s, g in zip(centres, rng.permutation(np.arange(k) % N_GROUPS))]
    return pd.DataFrame(rows, columns=["chrom", "centre", "group"])


def _genome(rng, hot: pd.DataFrame) -> dict[str, str]:
    seqs = {c: rng.choice(np.frombuffer(b"ACGT", "S1"), n) for c, n in CHROM_SIZES.items()}
    for h in hot.itertuples():
        m = np.frombuffer(MOTIFS[f"M{h.group}"].encode(), "S1")
        s = h.centre - len(m) // 2
        seqs[h.chrom][s:s + len(m)] = m
    return {c: s.tobytes().decode() for c, s in seqs.items()}


def _fragments(rng, hot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in range(N_GROUPS):
        mine = hot[hot.group == g]
        for i in range(CELLS_PER_GROUP):
            n = int(rng.integers(800, 3_000))
            in_hot = rng.random(n) < 0.5
            pick = mine.iloc[rng.integers(len(mine), size=n)]
            chrom = np.where(in_hot, pick.chrom.to_numpy(),
                             rng.choice(list(CHROM_SIZES), n, p=[0.45, 0.35, 0.2]))
            size = np.array([CHROM_SIZES[c] for c in chrom])
            length = rng.integers(40, 400, n)
            start = np.where(in_hot, pick.centre.to_numpy() - length // 2 + rng.integers(-150, 150, n),
                             (rng.random(n) * (size - 2_000)).astype(np.int64))
            rows.append(pd.DataFrame({"chrom": chrom, "start": start, "end": start + length,
                                      "barcode": f"g{g}c{i:03d}-1", "count": 1}))
    df = pd.concat(rows)
    return df.sort_values(["chrom", "start", "end", "barcode"]).drop_duplicates(
        ["chrom", "start", "end", "barcode"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    """Paths and ground truth: ``fasta``, ``meme``, ``parquet``, ``tsv``, ``fragments``, ``hotspots``."""
    d = tmp_path_factory.mktemp("world")
    rng = np.random.default_rng(0)
    hot = _hotspots(rng)

    fasta = d / "genome.fa"
    with open(fasta, "w") as fh:
        for c, seq in _genome(rng, hot).items():
            fh.write(f">{c}\n" + "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")

    meme = d / "motifs.meme"
    with open(meme, "w") as fh:
        fh.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
                 "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for name, seq in MOTIFS.items():
            fh.write(f"MOTIF {name} {name}\nletter-probability matrix: alength= 4 w= {len(seq)} nsites= 20 E= 0\n")
            for b in seq:
                fh.write(" " + " ".join("0.94" if b == x else "0.02" for x in "ACGT") + "\n")
            fh.write("\n")

    df = _fragments(rng, hot)
    tsv = d / "frags.tsv.gz"
    with gzip.open(tsv, "wt") as fh:
        df.to_csv(fh, sep="\t", header=False, index=False)
    parquet = ga.io.make_parquet(tsv, d / "frags.parquet")
    return {"dir": d, "fasta": fasta, "meme": meme, "tsv": tsv, "parquet": parquet,
            "fragments": df, "hotspots": hot}


def _group(a: ad.AnnData) -> pd.Categorical:
    return pd.Categorical(a.obs_names.str.slice(0, 2))


@pytest.fixture(scope="session")
def _tiles(world):
    a = ga.pp.make_tile_matrix(world["parquet"], chrom_sizes=CHROM_SIZES, tile_size=5_000,
                               min_fragments_per_cell=100)
    a.obs["group"] = _group(a)
    return a


@pytest.fixture
def tiles(_tiles):
    """Cell × 5 kb tile matrix (counts), with the true group in ``obs["group"]``."""
    return _tiles.copy()


@pytest.fixture(scope="session")
def _peaks(_tiles, world):
    a = _tiles.copy()
    ga.peak.call(a, "group", world["parquet"], genome=CHROM_SIZES, verbose=False)
    ga.peak.merge(a, chrom_sizes=CHROM_SIZES)
    p = ga.peak.make_matrix(a, world["parquet"], genome=CHROM_SIZES, verbose=False)
    p.obs["group"] = _group(p)
    p.uns["gmacs"] = a.uns["gmacs"]
    return p


@pytest.fixture
def peaks(_peaks):
    """Cell × merged-peak matrix built through call → merge → make_matrix, per-group peaks in ``uns["gmacs"]``."""
    return _peaks.copy()
