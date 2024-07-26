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

seq2PRINT: step 1 (create model training configuration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.seq_model_config
   tl.seq_lora_model_config
   tl.seq_lora_slice_model_config

seq2PRINT: step 2 (launch training scripts for the models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.launch_seq2print


seq2PRINT: step 3 (generating the sequence-based TF binding scores)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.seq_tfbs_seq2print
   tl.seq_attr_seq2print

seq2PRINT: step 4 (de novo motif discovery)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.seq_denovo_seq2print
   tl.delta_effects_seq2print
   tl.modisco_report
