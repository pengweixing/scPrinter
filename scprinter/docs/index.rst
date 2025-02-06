scprinter: A Python package for single-cell multi-scale footprinting
=====================================================================

scPrinter is the python implementation of PRINT.

Main advantages:
 - Much faster (40-60x faster)
 - More flexible (generate footprints for any region / cell groupings)
 - More portable (No need to share fragments, share the printer object (10x smaller))
 - Real time synced visualization of footprinting results.
 - Generalizable to low-cov bulk ATAC & scATAC-seq

Coming functions:
 - Differential testing



.. video:: _static/beta.mp4
    :autoplay:
    :loop:
    :width: 960


How to cite
-----------
The original multi-scale footprinting algorithm is described in the following paper:

- Hu, Yan, et al. "Single-cell multi-scale footprinting reveals the modular organization of DNA regulatory elements." bioRxiv (2023): 2023-03.

.. code-block::

  @article{hu2025multiscale,
  title={Multiscale footprints reveal the organization of cis-regulatory elements},
  author={Hu, Yan and Horlbeck, Max A and Zhang, Ruochi and Ma, Sai and Shrestha, Rojesh and Kartha, Vinay K and Duarte, Fabiana M and Hock, Conrad and Savage, Rachel E and Labade, Ajay and others},
  journal={Nature},
  pages={1--8},
  year={2025},
  publisher={Nature Publishing Group UK London}
}


.. toctree::
   :maxdepth: 3
   :hidden:

   tutorials/index
   reference/index
