# Contributing

Thank you for your interest in contributing to GATAC!

---

## Development setup

```bash
git clone https://github.com/Soldatov-Lab/gatac.git
cd GATAC

# Install in editable mode with all dev extras
uv sync --extra cuda12
```

---

## Running tests

Tests are located in `tests/` (unit tests) and in the
[`reproducibility/`](https://github.com/Soldatov-Lab/gatac-reproducibilty)
submodule (tracked at `reproducibility/`) for integration/benchmark tests.

After cloning, initialise the submodule once:

```bash
git submodule update --init --recursive
```

```bash
# Unit tests
uv run pytest tests/

# Reproducibility benchmarks (requires GPU + data)
cd reproducibility
pixi run pytest test/
```

After any change to a preprocessing or analysis function, run the related
reproducibility test:

```bash
cd reproducibility
pixi run pytest test/tile_matrix.py
pixi run pytest test/spectral_embedding.py
```

---

## Code style

- **Formatter**: `ruff format`
- **Linter**: `ruff check`
- **Type hints**: encouraged for all public functions
- **Docstrings**: NumPy style

```bash
uv run ruff format gatac/
uv run ruff check gatac/
```

---

## Submitting changes

1. Fork the repository and create a feature branch.
2. Make your changes with appropriate tests.
3. Ensure all existing tests pass.
4. Open a pull request with a clear description of the changes.

---

## Reporting issues

Please open a GitHub issue with:

- A minimal reproducible example
- GATAC version (`gatac --version`)
- GPU model and CUDA version
- Error traceback

---

## Documentation

The docs live in `docs/`.  To build locally:

```bash
uv sync --extra docs
cd docs
make html
# Open _build/html/index.html
```

Analysis notebooks live in the separate
[`gatac-notebooks`](https://github.com/Soldatov-Lab/gatac-notebooks)
repository — see [Tutorials](tutorials/index.md) for details.
