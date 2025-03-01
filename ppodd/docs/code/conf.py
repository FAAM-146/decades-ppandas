# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_rtd_theme  # type: ignore
import datetime

from ppodd import version as ppodd_version, githash

# -- Project information -----------------------------------------------------

project = "DECADES-(PP)andas Post-Processing Suite"
copyright = f"{datetime.datetime.now().year}, FAAM"
author = "FAAM"
release = f"{ppodd_version()} ({githash()})"
version = ppodd_version()


# -- Path setup --------------------------------------------------------------


def setup(app) -> None:
    app.add_css_file("mods.css")


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx_rtd_theme", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]

# Include both the class docstring and the __init__ docstring
napoleon_include_init_with_doc = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "base_rst"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]
# latex_toplevel_sectioning = 'section'
latex_elements = {
    "papersize": "a4paper",
    "extraclassoptions": "openany,oneside",
}
latex_logo = "../static/faam.png"
