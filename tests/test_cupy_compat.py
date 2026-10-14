"""bincount workaround for the CuPy 14 CUB histogram crash."""
import cupy as cp
import numpy as np
import pytest

from gatac._cupy_compat import bincount


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
