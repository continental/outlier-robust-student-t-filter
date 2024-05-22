Welcome to the API to the Student-t Bayesian Filtering implementations!
=======================================================================

This package contains next to the basic implementation of the Student-t Filter 
:class:`~src.filters.proposed.StudentTFilter`, a more complete framework to 
compare different state-of-the-art methods.

This documentation can be recompiled using the command ``make html`` in the ``doc/`` directory.

Framework Documentation
=======================

The documentation for the framework can be found in

.. autosummary:: 
   :toctree: generated
   :recursive:
   :caption: Contents:

   src.filters
   src.eval
   src.visual
   src.utils

Special Scripts and Protocols
=============================

| To reproduce our results look into the Jupyter Notebook ``scripts/quantitative_results.ipynb``.
| Since the framework is rather involved, we also provide a bare-bones implementation of the principles in our framework in ``minimalCodeExample.ipynb``.
| For interest in more qualitative results, we provide an interactive demonstrator to showcase influences of different parameters at ``scripts/showcase.ipynb``.
| And the figures in the paper were created with ``scripts/Figures.ipynb``.



Indices and tables
==================

* :ref:`genindex`

.. * :ref:`search`
.. We can include jupyter notebooks into the API by converting a .ipynb file into a .py file which is executed to produce the page. Converting is done via
.. 'jupytext --update-metadata '{"jupytext": {"cell_markers": "\"\"\""}}' --to py:percent <your-new-file>.ipynb'
.. see 'https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion/blob/ab1df923b02a7add6f78ecbf5f935b48930f80cd/docs/notebooks/intro.ipynb' for an example
