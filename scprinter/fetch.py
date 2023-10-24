from .utils import *
import numpy as np

# get ATAC insertions for a bunch of cells and a given reigon
def _get_region_atac(printer,
                  cell_barcodes, region):
    insertion_profile = printer.fetch_insertion_profile()
    v = region.values.reshape((-1))
    chrom, start_pos, end_pos = v[0], v[1], v[2]
    cell_index = cell_barcodes

    v = insertion_profile[chrom][cell_index, start_pos:end_pos]
    return v

# A wrapper function for more complicated situations
# For instance, cell_barcodes are string not index
# region are "chr1:1-10000"
def get_region_atac(printer,
                  cell_barcodes,
                  region):
    region = regionparser(region, printer)
    cell_index = printer.insertion_file.obs_ix(np.array(cell_barcodes))

    return _get_region_atac(printer, cell_index, region)

# get ATAC insertion for groups and a given region
def _get_group_atac(printer,
                 cell_grouping_idx, region):
    # why slice(None)?
    # Because insertion files are csc_matrix, they are fast when doing col slice
    # But not so when row selection, so we would try to minimize such effect.
    atac_all = _get_region_atac(printer, slice(None), region)
    a = np.array([np.asarray(atac_all[barcodes].sum(axis=0)).astype('float32')[0] for barcodes in cell_grouping_idx])

    return a

# A wrapper function for more complicated situations
def get_group_atac(printer,
                   cell_grouping, region):
    region = regionparser(region, printer)
    cell_grouping_idx = cell_grouping2cell_grouping_idx(printer, cell_grouping)

    return _get_group_atac(printer, cell_grouping_idx, region)