# Minimal makefile so you can do `make html` etc.
# `sphinx-build` is put by Sphinx into the SPHINXBUILD variable.
# SOURCEDIR is the repo root so the `notebooks/` submodule is reachable.
# CONFDIR keeps conf.py in this (docs/) directory.

SPHINXOPTS    ?=
SPHINXBUILD   ?= uv run sphinx-build
SOURCEDIR     = ..
CONFDIR       = .
BUILDDIR      = _build

.PHONY: help html clean livehtml

help:
	@$(SPHINXBUILD) --help

html:
	@$(SPHINXBUILD) -c "$(CONFDIR)" -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

livehtml:
	uv run sphinx-autobuild -c "$(CONFDIR)" "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"

.PHONY: Makefile
%: Makefile
	@$(SPHINXBUILD) -c "$(CONFDIR)" -b $@ "$(SOURCEDIR)" "$(BUILDDIR)/$@" $(SPHINXOPTS) $(O)
