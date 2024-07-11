===========
Tools: `tl`
===========
.. currentmodule:: scprinter

A collection of tools for footprinting, binding score calculation (model II), and differential testing etc.
The difference between `preprocessing (pp)` and `tools (tl)` is that `pp`
usually involves translating data from one format to another (fragments to insertion matrix, insertion to bigwig),
while `tl` is more about manipulating data into more insightful representations such as footprinting, binding score calculation,
differential testing.

Footprinting
~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.get_footprint_score
   tl.footprint_generator


Binding Score Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.get_binding_score

Insertion profile
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.get_insertions
