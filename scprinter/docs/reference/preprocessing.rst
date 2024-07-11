===================
Preprocessing: `pp`
===================
.. currentmodule:: scprinter

These are the function that are used to preprocess the data.
The difference between `preprocessing (pp)` and `tools (tl)` is that `pp`
usually involves translating data from one format to another (fragments to insertion matrix, insertion to bigwig),
while `tl` is more about manipulating data into more insightful representations such as footprinting, binding score calculation,
differential testing.

Fragment file processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    pp.import_fragments


Cell x Peak matrix preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    pp.make_peak_matrix
    pp.make_gene_matrix
    pp.mean_norm_counts

Export pseudobulk bigwigs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    pp.export_bigwigs
