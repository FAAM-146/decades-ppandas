# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

PYTHON        = python
PREPROCESSOR  = preproc.py

PUBHOST		  = faam-webserver
PUBDIR        = /var/www/html/additional/sphinx/coredata

RM            = /bin/rm

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

publish:
	make html
	ssh $(PUBHOST) mkdir -p $(PUBDIR)
	rsync -a _build/html/ $(PUBHOST):$(PUBDIR)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(PYTHON) "$(PREPROCESSOR)" "$@"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@$(RM) dynamic_content/[a-zA-Z]*
