# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html



# Monkey-patch autosummary template context, see https://stackoverflow.com/a/72658470
from sphinx.ext.autosummary.generate import AutosummaryRenderer

def smart_fullname(fullname):
    #return ".".join(fullname.split(".")[1:])
    return fullname.split(".")[-1]

def fixed_init(self, app):
    AutosummaryRenderer.__old_init__(self, app)
    self.env.filters["smart_fullname"] = smart_fullname

  
AutosummaryRenderer.__old_init__ = AutosummaryRenderer.__init__
AutosummaryRenderer.__init__ = fixed_init



print("- Loading documentation config.py file")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Student-t Bayesian Filtering'
copyright = '2024 co-pace GmbH (a subsidiary of Continental AG)'
author = 'Jonas Neuhöfer'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 
              'sphinx.ext.autosummary',
              "sphinx.ext.autosectionlabel",
              'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              ]
autosummary_generate = True
autosectionlabel_prefix_document = True

templates_path = ['_templates']
exclude_patterns = []

add_module_names = False
master_doc = 'index'
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False,
    'navigation_with_keys': True,
}
html_sidebars = {
    '**': ['globaltoc.html', 'localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'],
}
# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True
# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True
# If true, the _copyright_ is shown in the HTML footer. Default is True.
html_show_copyright = True

# Adding the sourcecode directory to the sys.path variable
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import src
