"""bincount workaround for the CuPy 14 CUB histogram crash, and the chunked host -> device copy."""
import ast
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

import gatac
import gatac._cupy_compat as compat
from gatac._cupy_compat import bincount, to_device


def test_no_unweighted_cupy_bincount():
    # Unweighted cp.bincount takes the crashing CUB path; gatac code calls the wrapper instead
    found = []
    for path in Path(gatac.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "bincount" and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ("cp", "cupy", "xp")
                    and not any(k.arg == "weights" for k in node.keywords)):
                found.append(f"{path.name}:{node.lineno}")
    assert not found, f"use gatac._cupy_compat.bincount: {found}"


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_bincount_genome_scale(dtype):
    # 6.2M bins (a genome-wide 500 bp tile matrix) crashes cp.bincount's CUB path in CuPy 14.0.1
    rng = np.random.default_rng(0)
    x = rng.integers(0, 6_200_000, 10_000_000).astype(dtype)
    got = bincount(cp.asarray(x), minlength=6_300_000)
    assert got.dtype == cp.int64
    assert np.array_equal(got.get(), np.bincount(x, minlength=6_300_000))


def test_bincount_edge_cases():
    assert bincount(cp.zeros(0, cp.int32), minlength=3).get().tolist() == [0, 0, 0]
    assert bincount(cp.asarray([4, 1, 1], cp.int32)).get().tolist() == [0, 2, 0, 0, 1]
    assert bincount(cp.asarray([True, False, True])).get().tolist() == [1, 2]
    with pytest.raises(ValueError):
        bincount(cp.asarray([-1, 2]))
    with pytest.raises(TypeError):
        bincount(cp.asarray([1.0]))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.int64, np.bool_])
def test_to_device_matches_asarray(dtype, monkeypatch):
    # a small chunk size exercises the chunked path, including a partial last chunk
    monkeypatch.setattr(compat, "_H2D_CHUNK", 1000)
    rng = np.random.default_rng(1)
    for shape in [(0,), (7,), (12_345,), (301, 7)]:
        h = (rng.random(shape) * 1000).astype(dtype)
        d = to_device(h)
        assert d.dtype == h.dtype and d.shape == h.shape
        assert np.array_equal(d.get(), h)


def test_to_device_views_casts_and_device_input(monkeypatch):
    monkeypatch.setattr(compat, "_H2D_CHUNK", 1000)
    h = np.arange(20_000, dtype=np.int32).reshape(200, 100)
    assert np.array_equal(to_device(h[:, ::3]).get(), h[:, ::3])        # non-contiguous view
    d = to_device(h, cp.int64)
    assert d.dtype == cp.int64 and np.array_equal(d.get(), h)
    x = cp.arange(5)
    assert to_device(x) is x
    assert to_device(x, cp.float32).dtype == cp.float32
