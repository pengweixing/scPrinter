========
Utils
========

.. currentmodule:: scprinter

These functions provides some handy tools for handling bed files, footprints data.


Genomic Ranges related functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    utils.regionparser
    utils.resize_bed_df
    utils.merge_overlapping_bed
    utils.df2regionidentifier

Cell grouping related functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    utils.cell_grouping2cell_grouping_idx
    utils.df2cell_grouping

Genome / DNA sequence related functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    utils.DNA_one_hot
    utils.GC_content
    utils.get_peak_bias
    utils.get_stats_for_genome

Footprinting post processing functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    utils.zscore2pval
