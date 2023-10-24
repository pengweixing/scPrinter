from .utils import *


def strided_axis0(a, L):
    # Store the shape and strides info
    shp = a.shape
    s  = a.strides

    # Compute length of output array along the first axis
    nd0 = shp[0]-L+1

    # Setup shape and strides for use with np.lib.stride_tricks.as_strided
    # and get (n+1) dim output array
    shp_in = (nd0,L)+shp[1:]
    strd_in = (s[0],) + s
    return np.lib.stride_tricks.as_strided(a, shape=shp_in, strides=strd_in)


def getPrecomputedBias(precomputed_bias_path,
                       regions,
                       savePath=None):
    regions = regionparser(regions)
    chrom_list, start_list, end_list = np.array(regions['Chromosome']), \
        np.array(regions['Start']).astype('int'), \
        np.array(regions['End']).astype('int')
    # print (chrom_list)
    uniq_chrom = np.unique(chrom_list)
    width = end_list[0] - start_list[0]
    with h5py.File(precomputed_bias_path, 'r') as dct:
        precomputed_bias = {chrom : np.array(dct[chrom]) for chrom in uniq_chrom}
    # Create the empty np.ndarray for final results
    final_result = np.zeros((len(regions), width))
    for chrom in uniq_chrom:
        bias = precomputed_bias[chrom]
        mask = chrom_list == chrom

        bias_region_chrom = strided_axis0(bias, width)[start_list[mask]]
        final_result[mask] = bias_region_chrom
    if savePath is not None:
        np.save(savePath, final_result)
    else:
        return final_result




