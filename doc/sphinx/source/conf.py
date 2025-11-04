import datetime

import shamrock

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

project = "python bindings"
copyright = f"2020 -- {datetime.datetime.now().year} Timothee David--Cléris"
author = "Timothee David--Cléris"

release = shamrock.version_string()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # For documenting Python code
    "sphinx.ext.intersphinx",  # to have link in source code
    # 'sphinx.ext.viewcode',  # For linking to the source code in the docs
    # sadly this does not seems to work as it expect real python sources which do not exist here
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    "sphinx_gallery.gen_gallery",  # generate thumbnail and example lib
    "sphinx_copybutton",  # add a copy button to code blocks
    "sphinx_design",  # Add grid tabs and fancy html stuff
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


intersphinx_mapping = {
    "ipykernel": ("https://ipykernel.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pip": ("https://pip.pypa.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

sphinx_gallery_conf = {
    "backreferences_dir": "_as_gen",  # link to source from examples
    "doc_module": ("shamrock"),  # The name of the module that is documented
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "_as_gen",  # path to where to save gallery generated output
    "line_numbers": True,  # line numbers in examples
    # The 3 next args are a bit like dark magic which allows the link
    # to functions in the example to exist
    "reference_url": {"shamrock": None},
    "prefer_full_module": {r"shamrock\."},
    "remove_config_comments": True,
    "filename_pattern": "/run_",  # Run all examples that start with run_
    "write_computation_times": True,  # write sg_execution_times
    "show_memory": True,
    # Capture matplotlib anim in examples
    "matplotlib_animations": True,
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "_static/large-figures/figures/no_background_nocolor.png"
# html_favicon = "_static/logo.png"
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
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Shamrock-code/Shamrock",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "Website",
            "url": "https://shamrock-code.github.io/",
            "icon": "_static/large-figures/figures/logo.png",
            "type": "local",
        },
    ],
}

html_context = {
    "github_url": "https://github.com",  # or your GitHub Enterprise site
    "github_user": "Shamrock-code",
    "github_repo": "Shamrock",
    "github_version": "main",
    "doc_path": "doc/sphinx/source",
}

html_css_files = [
    "css/custom.css",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
