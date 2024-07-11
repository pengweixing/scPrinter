===========
Buencolors
===========
.. currentmodule:: scprinter

The BuenColors in Python! Credits to Caleb Lareau: https://github.com/caleblareau/BuenColors for the R version

Example usage
~~~~~~~~~~~~~

```
discrete_palette = jdb_palette("GrandBudapest", 4, 'discrete')
continuous_palette = jdb_palette("GrandBudapest", 10, 'continuous')
```

For seaborn, directly use the generated list for discrete palettes.
For matplotlib, use LinearSegmentedColormap.from_list(name, continuous_palette) for continuous palettes.



Buencolors
~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   buencolors.jdb_palette
