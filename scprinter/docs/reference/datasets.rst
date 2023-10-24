========
Datasets
========
.. currentmodule:: scprinter

These functions provides an easy access to genome specific public datasets
and footprinting related datasets used in scprinter.


.. note::

    By default it will save the file to `~/.cache/scprinter_cache`
    But you can overwrite it with the environment variable `SCPRINTER_DATA`

Pretrained Models
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    datasets.dispersion_model
    datasets.TFBS_model
    datasets.TFBS_model_classI
    datasets.NucBS_model


Genome
~~~~~~

.. autosummary::
    :toctree: _autosummary

    genome.Genome
    genome.GRCh38
    genome.GRCm38
    genome.hg38
    genome.mm10

Tutorial Datasets
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    datasets.BMMCTutorial

