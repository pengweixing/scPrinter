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

Beta testing on fasrc!
----------------------
**Why Fasrc?**

That way the lab can share one environment and we can all use the same version of the software.
If people raise issues, I can fix them and avoid people reinstalling softwares.

How to cite
-----------
The original multi-scale footprinting algorithm is described in the following paper:

- Hu, Yan, et al. "Single-cell multi-scale footprinting reveals the modular organization of DNA regulatory elements." bioRxiv (2023): 2023-03.

.. code-block::

  @article{hu2023single,
  title={Single-cell multi-scale footprinting reveals the modular organization of DNA regulatory elements},
  author={Hu, Yan and Ma, Sai and Kartha, Vinay K and Duarte, Fabiana M and Horlbeck, Max and Zhang, Ruochi and Shrestha, Rojesh and Labade, Ajay and Kletzien, Heidi and Meliki, Alia and others},
  journal={bioRxiv},
  pages={2023--03},
  year={2023},
  publisher={Cold Spring Harbor Laboratory} }


.. toctree::
   :maxdepth: 3
   :hidden:

   install
   tutorials/index
   reference/index
