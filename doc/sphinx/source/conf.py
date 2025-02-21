# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'python bindings'
copyright = '2020 -- Timothee David--Cléris'
author = 'Timothee David--Cléris'

# The full version, including alpha/beta/rc tags
release = '2024.10.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # For documenting Python code
    'sphinx.ext.viewcode',  # For linking to the source code in the docs
    'sphinx.ext.autosummary',  # Create neat summary tables for modules/classes/methods etc
    'sphinx_gallery.gen_gallery', # generate thumbnail and exemple lib
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

sphinx_gallery_conf = {
    'backreferences_dir': '_gen_exemples/backreferences', # link to source from exemples
    'doc_module': ('shamrock'),
    'examples_dirs': '../exemples',   # path to your example scripts
    'gallery_dirs': '_gen_exemples',  # path to where to save gallery generated output
    'line_numbers': True,
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
html_logo = "_static/large-figures/figures/no_background_nocolor.png"
#html_favicon = "_static/logo.png"
html_sourcelink_suffix = ""
html_last_updated_fmt = ""  # to reveal the build date in the pages meta

html_theme_options = {
    "logo": {
        "text": project,
        "image_dark": html_logo,
    },
    "use_edit_page_button": True,
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "search_as_you_type": True,
}

html_context = {
    "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "tdavidcl",
    "github_repo": "Shamrock",
    "github_version": "main",
    "doc_path": "doc/sphinx/source",
}

html_css_files = [
    'css/custom.css',
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
