=============
API reference
=============

An overview of scprinter API.

Some commonly used arguments in scprinter:

- `cell_grouping` : list[list[str]] | list[str] | np.ndarray
    Essentially, pseudo-bulks, specifiec by a list of the cell barcodes belong to this group, e.g.
        `['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112', 'ACAGTGGT,ACAGTGGT,ATCACGTT,BUENSS112', 'ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']`.  If you want to visualize multiple groups, you can provide a list of lists, e.g.
        `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`.
- `group_names`: list[str] | str
    The names of the groups, e.g. `['Group1', 'Group2']` It needs to have the same length as the `cell_grouping` list.
- `save_key`: str
    If you generate footprints / TF binding score for multiple region / groups, you can specify a key to save the results in the printer object or local path. `save_key` refers to the collection of these results
- `save_path`: str
    The path to save the results. It needs to contain the file name as well, such as `/data/rzhang/modisco.h5`
- `wandb_project`: str
    The wandb project name to log the training process. If you don't want to log the training process, you can set it to `None`. But I highly recommend you to log the training process, so that you can track the training process and compare the results across different runs.
    Check https://wandb.ai/home for more details.

.. toctree::
   :maxdepth: 2

   io
   preprocessing
   tools
   plotting
   motifs
   datasets
   peak
   chromvar
   buencolors
   utils
   seq
