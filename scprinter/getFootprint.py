import os
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["NUMEXPR_MAX_THREADS"] = '1'
import scipy
from scipy.ndimage import maximum_filter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import trange, tqdm
import torch.nn.functional as F
import warnings
from .utils import *

# Hard-coded these, but for better understanding
position_key, group_key, count_key = 0, 1, 2

# Retrieve the pseudobulk-by-position ATAC insertion matrix for a specific genomic regoin
def getRegionATAC(select,  # np array of (, 3): position, group, count
                  n_groups,  # Number of the unique group
                  width,  # Width of the region
                  select_group=None,
                  ):
    # select can be a handler in hdf5 file, so need np.array to actually read it
    select = np.array(select)

    if select_group is not None:
        mask = np.isin(select[:, group_key], select_group)
    else:
        mask = slice(None)

    matrix = np.zeros((n_groups, width))
    matrix[select[mask, group_key], select[mask, position_key]] = select[mask, count_key]
    return matrix


# Doing the same thing as conv in R, but more generalizable
def rz_conv(a, n=2):
    # a can be shape of (batch, sample,...,  x) and x will be the dim to be conv on
    # pad first:
    shapes = np.array(a.shape)
    shapes[-1] = n
    a = np.concatenate([np.zeros(shapes), a, np.zeros(shapes)], axis=-1)
    ret = np.cumsum(a, axis=-1)
    # ret[..., n * 2:] = ret[..., n * 2:] - ret[..., :-n * 2]
    # ret = ret[..., n * 2:]
    ret =  ret[..., n * 2:] - ret[..., :-n * 2]
    return ret


# Given a value matrix x, along the last dim,
# for every single position, calculate sum of values in center and flanking window
# Essentially we are doing a running-window sum for the center and flanking
def footprintWindowSum(x,  # A numerical or integer vector x
                       footprintRadius,  # Radius of the footprint region
                       flankRadius  # Radius of the flanking region (not including the footprint region)
                       ):
    halfFlankRadius = int(flankRadius / 2)
    width = x.shape[-1]

    # Calculate sum of x in the left flanking window
    shift = halfFlankRadius + footprintRadius
    # x can be shape of (x) or (sample, x)
    shapes = np.array(x.shape)
    shapes[-1] = shift
    leftShifted = np.concatenate([np.zeros(shapes), x], axis=-1)

    leftFlankSum = rz_conv(leftShifted, halfFlankRadius)[..., :width]

    # Calculate sum of x in the right flanking window
    rightShifted = np.concatenate([x, np.zeros(shapes)], axis=-1)
    rightFlankSum = rz_conv(rightShifted, halfFlankRadius)[..., shift:]

    centerSum = rz_conv(x, footprintRadius)
    return {'leftFlank': leftFlankSum,
            'center': centerSum,
            'rightFlank': rightFlankSum}


# Calculate footprint score track for a single region and multiple samples
def footprintScoring_multi(Tn5Insertion,
                           # Integer vector of raw observed Tn5 insertion counts at each single base pair
                           Tn5Bias,  # Vector of predicted Tn5 bias. Should be the same length
                           dispersionModel,
                           # Background dispersion model for center-vs-(center + flank) ratio insertion ratio
                           footprintRadius=10,  # Radius of the footprint region
                           flankRadius=10  # Radius of the flanking region (not including the footprint region)
                           ):
    modelWeights = dispersionModel['modelWeights']

    # Get sum of predicted bias in left flanking, center, and right flanking windows
    biasWindowSums = footprintWindowSum(Tn5Bias,
                                        footprintRadius,
                                        flankRadius)

    # Get sum of insertion counts in left flanking, center, and right flanking windows
    insertionWindowSums = footprintWindowSum(Tn5Insertion,
                                             footprintRadius,
                                             flankRadius)
    leftTotalInsertion = insertionWindowSums['center'] + insertionWindowSums['leftFlank']
    rightTotalInsertion = insertionWindowSums['center'] + insertionWindowSums['rightFlank']
    # Prepare input data (foreground features) for the dispersion model
    try:
        fgFeatures = np.stack([
            np.array([biasWindowSums['leftFlank']] * len(Tn5Insertion)),
            np.array([biasWindowSums['rightFlank']] * len(Tn5Insertion)),
            np.array([biasWindowSums['center']] * len(Tn5Insertion)),
            np.log10(leftTotalInsertion),
            np.log10(rightTotalInsertion)
        ], axis=-1)
    except:
        for xx in [
            np.array([biasWindowSums['leftFlank']] * len(Tn5Insertion)),
            np.array([biasWindowSums['rightFlank']] * len(Tn5Insertion)),
            np.array([biasWindowSums['center']] * len(Tn5Insertion)),
            np.log10(leftTotalInsertion),
            np.log10(rightTotalInsertion)
        ]:
            print (xx.shape)
            raise EOFError
    fgFeaturesScaled = (fgFeatures - dispersionModel['featureMean']) / (dispersionModel['featureSD'])
    # Given observed center bias, flank bias, and total insertion, use our model to estimate background
    # dispersion of and background mean of center-vs-(center + flank) ratio
    with torch.no_grad():
        predDispersion = predictDispersion_jit(torch.from_numpy(fgFeaturesScaled).float(), *modelWeights).numpy()
    # predDispersion = predictDispersion(fgFeaturesScaled, modelWeights)
    predDispersion = predDispersion * dispersionModel['targetSD']
    predDispersion = predDispersion + dispersionModel['targetMean']


    leftPredRatioMean = predDispersion[..., 0] #- predDispersion2[..., 0]
    leftPredRatioSD = predDispersion[..., 1] #- predDispersion2[..., 1]
    rightPredRatioMean = predDispersion[..., 2] #- predDispersion2[..., 2]
    rightPredRatioSD = predDispersion[..., 3] #- predDispersion2[..., 3]

    # Calculate foreground (observed) center-vs-(center + flank) ratio
    fgLeftRatio = insertionWindowSums['center'] / leftTotalInsertion
    fgRightRatio = insertionWindowSums['center'] / rightTotalInsertion

    # Compute p-value based on background mean and dispersion
    leftPval = scipy.stats.norm.cdf(fgLeftRatio, leftPredRatioMean, leftPredRatioSD) #/ scipy.stats.norm.cdf(fgLeftRatioBias, leftPredRatioMean2, leftPredRatioSD2)

    # This is to make it consistent with R pnorm
    leftPval[np.isnan(leftPval)] = 1
    rightPval = scipy.stats.norm.cdf(fgRightRatio, rightPredRatioMean, rightPredRatioSD) #/ scipy.stats.norm.cdf(fgRightRatioBias, rightPredRatioMean2, rightPredRatioSD2)

    rightPval[np.isnan(rightPval)] = 1

    # Combine test results for left flank and right flank by taking the bigger pval
    p = np.maximum(leftPval, rightPval)

    # Mask positions with zero coverage on either flanking side
    p[(leftTotalInsertion < 1) | (rightTotalInsertion < 1)] = 1
    return p

# numpy version
def predictDispersion(x, modelWeights):
    x = x @ modelWeights[0].T.cpu().numpy() + modelWeights[1].cpu().numpy()
    x = np.clip(x, 0.0, a_max=np.Inf)
    x = x @ modelWeights[2].T.cpu().numpy() + modelWeights[3].cpu().numpy()
    return x

# torch version
def predictDispersion_torch(x, modelWeights):
    with torch.no_grad():
        x = F.relu(F.linear(torch.from_numpy(x).float(), modelWeights[0], modelWeights[1]))
        x = F.linear(x, modelWeights[2], modelWeights[3])
        return x.detach().cpu()

# torch jit version
@torch.jit.script
def predictDispersion_jit(x, modelWeights_0,
                            modelWeights_1,
                          modelWeights_2,
                          modelWeights_3):
    x = F.relu(F.linear(x, modelWeights_0, modelWeights_1))
    x = F.linear(x, modelWeights_2, modelWeights_3)
    return x

# Calculate footprint score track for a single genomic region
def regionFootprintScore(regionATAC_param,
                         Tn5Bias,
                         dispersionModel,
                         footprintRadius,
                         flankRadius,
                         extra_info, # extra_info to be returned, so the parent process would know which child it is.
                        smoothRadius=None
                         ):
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        if type(regionATAC_param) is tuple:
            regionATAC = getRegionATAC(*regionATAC_param)
        else:
            regionATAC = regionATAC_param

        # Calculate the pseudo-bulk-by-position footprint pvalue matrix
        footprintPvalMatrix = footprintScoring_multi(
            Tn5Insertion=regionATAC,
            Tn5Bias=Tn5Bias,
            dispersionModel=dispersionModel,
            footprintRadius=footprintRadius,
            flankRadius=flankRadius
        )
        if smoothRadius is None:
            smoothRadius = int(footprintRadius / 2)
        footprintPvalMatrix[np.isnan(footprintPvalMatrix)] = 1 # Set NA values to be pvalue = 1
        # print (footprintPvalMatrix, np.sum(np.isnan(footprintPvalMatrix)), np.sum(np.isinf(footprintPvalMatrix)))
        pvalScoreMatrix = -np.log10(footprintPvalMatrix)
        pvalScoreMatrix[np.isnan(pvalScoreMatrix)] = 0
        pvalScoreMatrix[np.isinf(pvalScoreMatrix)] = 20
        maximum_filter_size = [0] * len(pvalScoreMatrix.shape)
        maximum_filter_size[-1] = 2 * smoothRadius
        pvalScoreMatrix = maximum_filter(pvalScoreMatrix, tuple(maximum_filter_size), origin=-1)
        # Changed to smoothRadius.
        pvalScoreMatrix = rz_conv(pvalScoreMatrix, smoothRadius) / (2 * smoothRadius)
        pvalScoreMatrix[np.isnan(pvalScoreMatrix)] = 0
        pvalScoreMatrix[np.isinf(pvalScoreMatrix)] = 20
    return pvalScoreMatrix, extra_info


def fastMultiScaleFootprints(region_ATAC,
                             Tn5Bias,
                             dispersionModels,
                             modes = np.arange(2, 101),
                             footprintRadius=None, # Radius of the footprint region
                             flankRadius=None, # Radius of the flanking region (not including the footprint region)
                             extra_info = None
                             ):
    return_array = None
    if footprintRadius is None:
        footprintRadius = modes
    if flankRadius is None:
        flankRadius = modes

    if type(region_ATAC) is tuple:
        region_ATAC = getRegionATAC(*region_ATAC)
    else:
        region_ATAC = region_ATAC
    modes = list(modes)

    for mode, r1, r2 in zip(modes, footprintRadius, flankRadius):
        result, mode = regionFootprintScore(
            region_ATAC,
            Tn5Bias,
            dispersionModels[str(mode)],
            r1,
            r2,
            mode,
            5,)
        # result, mode = p.result()
        # bar.update(1)
        if return_array is None:
            return_array = np.zeros((result.shape[0], len(modes), result.shape[-1]))
        return_array[:, modes.index(mode), :] = result

    if extra_info is not None:
        return return_array, extra_info
    return return_array



def oneScaleFootprint(CountTensorPath,
                        region_identifier_chunk,
                        width, num_groups, select_group_id, seqBiass,
                        dispModel, mode, footprintRadius, flankRadius):
    return_array = None
    with h5py.File(CountTensorPath, "r") as f:
        for i, region_identifier in enumerate(region_identifier_chunk):
            try:
                region_count = np.array(f[region_identifier])
            except:
                print (region_identifier, "has zero counts in the given groups")

            region_ATAC = getRegionATAC(region_count, num_groups, width, select_group_id)
            result, mode = regionFootprintScore(
                region_ATAC,
                seqBiass[i],
                dispModel,
                footprintRadius,
                flankRadius,
                mode,
                5, )

            if return_array is None:
                return_array = np.zeros((len(region_identifier_chunk),
                                         result.shape[0], result.shape[-1]))
            return_array[i, :,  :] = result



    return return_array, mode


# Calculate footprint score track for multiple regions but one scale.
def getFootprints(
                CountTensorPath,
                regions,
                groups,
                dispersionModels,
                modes, # int or list of int. This is used for retrieving the correct dispersion model.
                Tn5Bias,
                footprintRadius=None, # Radius of the footprint region
                flankRadius=None, # Radius of the flanking region (not including the footprint region)
                nCores = 16, # Number of cores to use
                saveDir = None,
                saveName = "chunkedFootprintResults.h5",
                verbose=True,
                returnCombined=None,
                append_mode=False, # When True, it means, the new results will be appended ore groups (mode 2)
        # / more regions (mode 1)), cannot have more reads, because that just changes the read count.
        # / more scales (mode 4), mode 3 is reserved for reads, but not supported here
                 ):
    if returnCombined is None:
        returnCombined = not (saveDir is None)
    assert append_mode != 3, "Don't support append fragments, need to be rerun!"
    if saveDir is not None:
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

    # we want a list of modes, but also compatible with one mode
    if type(modes) is int:
        modes = [modes]

    # If no radius provided, reuse the modes
    if footprintRadius is None:
        footprintRadius = modes
    if flankRadius is None:
        flankRadius = modes

    # Again, expct to see a list, if not, create one
    if type(footprintRadius) is int:
        footprintRadius = [footprintRadius]
    if type(flankRadius) is int:
        flankRadius = [flankRadius]

    regions = regionparser(regions)
    width = regions.iloc[0]['End'] - regions.iloc[0]['Start']

    if verbose:
        print ("identified region width: ", width)
        bar = trange(len(regions) * len(modes), desc=' - Footprinting!')

    with h5py.File(CountTensorPath, "r") as f:
        groups = [str(xx) for xx in groups]
        # region_identifier is the string like chr1:0-1000 that identifies each region.
        region_identifier = df2regionidentifier(regions)
        uniq_group, existing_counttensor_groups, group_notin_counttensor, \
            uniq_region, existing_region, region_notin_counttensor = Unify_meta_info(f,
                                                addition_feats=[groups,
                                                                region_identifier],
                                                entries=['group', 'regions'],
                                                dtypes=['str', 'str'])

        assert len(group_notin_counttensor) == 0 , "observe new groups that don't have count tensor: %s..." % group_notin_counttensor[0]
        assert len(region_notin_counttensor) == 0, "cannot have new regions that don't have count tensor: %s..." % region_notin_counttensor[0]

        group_map = {g:i for i,g in enumerate(existing_counttensor_groups)}
        select_group_id = np.array([group_map[g] for g in groups])
        if np.sum(select_group_id - np.arange(len(groups))) == 0:
            select_group_id = None

    if saveDir is not None:
        final_hdf5_path = os.path.join(saveDir, "%s" % saveName)
        final_hdf5 = h5py.File(final_hdf5_path, "a" if append_mode else "w")

        uniq_group, existing_group, new_group, \
        uniq_region, existing_region, new_region, \
        uniq_scale, existing_scale, new_scale = Unify_meta_info(final_hdf5,
                                            addition_feats=[groups, region_identifier, modes],
                                            entries=['group', 'regions', 'modes'],
                                            dtypes=['str', 'str', 'int'])

        if append_mode == 1:
            # append region mode, check if there are new groups:
            assert len(new_group) == 0, "append_region mode, but observed changed cell grouping: %s ..." % \
                                        new_group[0]
            assert len(new_scale) == 0, "append_region mode, but observed changed scales: %d ..." % new_scale[0]
            assert len(new_region) == len(uniq_region), "append mode, cannot have overlap regions: %s" % \
                                                        uniq_region[np.isin(uniq_region, existing_region)][0]
        elif append_mode == 2:
            # append group mode, check if there are new regions
            assert len(new_region) == 0, "append_group mode, but observed changed regions: %s..." % new_region[
                0]
            assert len(new_scale) == 0, "append_group mode, but observed changed scales: %d ..." % new_scale[0]
            assert len(new_group) == len(uniq_group), "append mode, cannot have overlap groups: %s" % \
                                                      uniq_group[np.isin(uniq_group, existing_group)][0]
        elif append_mode == 4:
            assert len(new_group) == 0, "append_scale mode, but observed changed cell grouping: %s ..." % \
                                        new_group[0]
            assert len(new_region) == 0, "append_scale mode, but observed changed regions: %s..." % new_region[
                0]
            assert len(new_scale) == len(uniq_scale), "append mode, cannot have overlap scale: %d" % uniq_scale[
                np.isin(uniq_scale, existing_scale)][0]



        if 'regions' in final_hdf5.keys():
            # delete existing groups and regions, since we'll rewrite later.
            del final_hdf5['regions'], final_hdf5['group']

        # hdf5 in pythonwants the dtype to be specific
        groups_all = np.array(list(existing_group) + list(uniq_group))
        groups_all = [str(xx) for xx in groups_all]

        final_hdf5.create_dataset('group', data=groups_all, dtype=h5py.special_dtype(vlen=str))
        regions_all = list(existing_region) + list(region_identifier)
        regions_all = sort_region_identifier(regions_all)
        final_hdf5.create_dataset('regions', data=regions_all, dtype=h5py.special_dtype(vlen=str))

        scales_all = np.concatenate([existing_scale, uniq_scale]).reshape((-1))
        # Making sure scales are always sorted
        # we store this sort_id, because it means later we can concat results,
        # and use this to get the correct order
        scale_sort_id = np.argsort(scales_all)
        scales_all = scales_all[scale_sort_id]
        final_hdf5.create_dataset('scales', data=scales_all)

        final_hdf5.attrs['description'] = 'stored value shape of (#group, #scale, #position)'


    small_chunk_size = max(int(8000 / (len(groups) / 2)), 100)
    print (small_chunk_size)
    pool = ProcessPoolExecutor(max_workers=nCores)
    modes = list(modes)
    final_return_array = None
    # dispersionModels = [dispersionModels[str(mode)] for mode in modes]
    for i in range(0, len(regions), small_chunk_size):
        slice_ = slice(i, i + small_chunk_size)
        p_list = []
        # pool on modes...
        for mode, r1, r2  in zip(modes, footprintRadius, flankRadius):
            p_list.append(pool.submit(
                oneScaleFootprint,
                CountTensorPath,
                region_identifier[slice_],
                width, len(uniq_group), select_group_id,
                Tn5Bias[slice_],
                dispersionModels[str(mode)],
                int(mode),  # int or list of int. This is used for retrieving the correct dispersion model.
                r1,
                r2
            ))


        return_array = [0] * len(modes)

        for p in as_completed(p_list):
            results, mode = p.result()
            return_array[modes.index(mode) ] = results
            if verbose:
                bar.update(len(results))

        return_array = np.stack(return_array, axis=2)

        if saveDir is not None:
            for result, regionID in zip(return_array, region_identifier[slice_]):
                xx = result
                if append_mode:
                    # If there's a region already, so must be append group or scale
                    if regionID in final_hdf5:
                        # shape of (# groups (old), # modes (old), # position)
                        old_result = np.array(final_hdf5[regionID])
                        if append_mode == 2:
                            # more groups? concat at the first dim
                            xx = np.concatenate([old_result, xx], axis=0)
                        elif append_mode == 4:
                            # more scales? need to recreate and refill.
                            xx = np.concatenate([old_result, xx], axis=1)
                            xx = xx[:, scale_sort_id]

                    # else: more regions? just create!

                final_hdf5.create_dataset(regionID, data=xx)

        if returnCombined:
            if final_return_array is None: final_return_array = np.zeros([len(regions)] + list(return_array.shape[1:]))
            final_return_array[slice_] = return_array


    if verbose:
        bar.close()
    pool.shutdown(wait=True)
    if saveDir is not None:
        final_hdf5.close()

    if returnCombined:
        return final_return_array


def appendFootprints_regions(*args,
                       **kwargs):
    getFootprints(*args, append_mode=1, **kwargs)


def appendFootprints_groups(*args, 
                       **kwargs):
    getFootprints(*args, append_mode=2, **kwargs)

def appendFootprints_modes(*args, **kwargs):
    getFootprints(*args, append_mode=4, **kwargs)