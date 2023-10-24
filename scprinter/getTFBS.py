import math
import os
import time
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["NUMEXPR_MAX_THREADS"] = '1'
from .getFootprint import *
from .utils import *

# torch version
def torch_predict_BindingScore(data, weights):
    data = torch.from_numpy(data).float()
    data = (data - weights['footprintMean']) / weights['footprintSd']

    weights = weights['weights']
    print (data.shape, weights[0].shape)
    if data.shape[-1] != weights[0].shape[1]:
        x = F.relu(F.linear(data, weights[0][:-1].T, weights[1] + weights[0][-1]))
    else:
        x = F.relu(F.linear(data, weights[0].T, weights[1]))
    x = F.relu(F.linear(x, weights[2].T, weights[3]))
    x = torch.sigmoid(F.linear(x, weights[4].T, weights[5]))
    return x.numpy()

# torch jit version
@torch.jit.script
def jit_predict_BindingScore(data, footprintMean, footprintSd,
                               weights_0, weights_1,
                               weights_2, weights_3,
                               weights_4, weights_5,):

    data = (data - footprintMean) / footprintSd

    if data.shape[-1] != weights_0.shape[0]:
        x = F.relu(F.linear(data, weights_0[:-1].T, weights_1 + weights_0[-1]))
    else:
        x = F.relu(F.linear(data, weights_0.T, weights_1))
    x = F.relu(F.linear(x, weights_2.T, weights_3))
    x = torch.sigmoid(F.linear(x, weights_4.T, weights_5))
    return x

# numpy version
def predict_BindingScore(data, weights):
    x = data @ weights[0] + weights[1]
    x = np.clip(x, 0.0, np.Inf)
    x = x @ weights[2] + weights[3]
    x = np.clip(x, 0.0, np.Inf)

    x = x @ weights[4] + weights[5]
    x = 1 / (1 + np.exp(-x))
    return x

# This is a fast trick to turn sth shape of (x,y,z) to (x,y,z-L+1, L)
# by tiling windows on the last axis.
def strided_lastaxis(a, L):
    s0, s1, s2 = a.strides
    l, m, n = a.shape
    return np.lib.stride_tricks.as_strided(a, shape=(l, m, n - L + 1, L), strides=(s0, s1, s2, s2))

# Get predicted TF binding scores for a specific region
def getRegionBindingScore(region_identifier, # an identifier for things that finished
                  regionATAC,  # Position-by-pseudobulk matrix of ATAC data for the current region
                  Tn5Bias,  # Numeric vector of predicted Tn5 bias for the current region
                  region,  # pyRanges object for the current region
                  dispModels,  # Background dispersion model for center-vs-(center + flank) ratio insertion ratio
                  BindingScoreModel,  # Model for predicting TF binding at any motif site
                  sites=None,  # pyRanges object. Genomic ranges of motif matches in the current region
                  # Must have $score attribute which indicates how well the motif is matched (between 0 and 1)
                  # If NULL, the region will be divided into equally sized tiles for TF binding scoring
                  tileSize=10,  # Size of tiles if sites is NULL
                  contextRadius=100  # Local radius of model input (in bp)
                  ):

    if sites is not None:
        # Only keep motif matches within the region
        # Although we specified that the input sites should already be matches in the current region
        # this step prevents erros in case the user for got to take the overlap
        # sites.intersect(region)
        # sites = sites.df
        sites = sites

    else:
        sites = region.window(tileSize)
        sites = sites.df
        # to make pyranges consistent with GRanges
        # Requires double check.
        # sites['End'] -= 1
        sites['End'].iloc[-1] += 1
        sites['Score'] = 1
        sites['TF'] = ''
        ## Yan, could you confirm if it's OK to do this?
        skip_site_check = False
    if len(sites) == 0:
        return {}
    if 'Strand' not in sites:
        sites['Strand'] = '*'
        skip_site_check = True
    else:
        skip_site_check = False

    width = len(Tn5Bias)
    scales = BindingScoreModel['scales']
    multiScaleFootprints = None

    start = time.time()
    if type(regionATAC) is tuple:
        regionATAC = getRegionATAC(*regionATAC)
    else:
        regionATAC = regionATAC

    for scale_index, scale in enumerate(scales):
        # I changed the code in regionFootprintscore to make it consistent
        footprintScores, _ = regionFootprintScore(regionATAC,
                                               Tn5Bias,
                                               dispModels[scale_index],
                                               scale,
                                               scale,
                                               None)
        if multiScaleFootprints is None:
            # shape of (group, scale, position)
            multiScaleFootprints = np.zeros((footprintScores.shape[0], len(scales), footprintScores.shape[1]),
                                            dtype='float32')
        multiScaleFootprints[:, scale_index, :] = footprintScores

    footprint_time = time.time() - start

    # Calculate positions of TF sites relative to the start of the CRE region
    relativePos = (np.array(sites['Start'] + sites['End']) * 0.5 - int(region.df['Start'].iloc[0])).astype(
        'int')
    # Only keep sites with distance to CRE edge >= contextRadius
    siteFilter = (relativePos > contextRadius) & (relativePos <= (width - contextRadius))
    sites = sites.iloc[siteFilter]
    relativePos = relativePos[siteFilter]

    # Go through each site and calculate predicted TF binding score for each pseudobulk
    score = np.array(sites['Score'])
    start = time.time()


    if len(sites) > 0:
        start_site = relativePos - contextRadius - 1
        slice_width = 2 * contextRadius + 1
        # This is the stride trick
        # shape of, (group, scale, sites, 2 * contextRadius + 1)
        siteFootprints = strided_lastaxis(multiScaleFootprints, slice_width)[:, :, start_site]

        if not skip_site_check:
            mask = np.isin(sites['Strand'], ['-', '-1', -1])
            siteFootprints[:, :, mask] = siteFootprints[:, :, mask][..., ::-1]

        stride_time = time.time() - start
        start = time.time()

        # shape of (group, sites, scale, 2 * context Radius + 1)
        # then reshaped to (group, sites, scale * (2 * context Radius + 1))
        BindingScoreData = np.transpose(siteFootprints, axes=[0, 2, 1, 3]).reshape((siteFootprints.shape[0],
                                                                                    siteFootprints.shape[2], -1))

        transpose_time = time.time() - start
        start = time.time()

        # score = torch_predict_BindingScore(BindingScoreData, BindingScoreModel)
        if 'model' in BindingScoreModel:
            score = BindingScoreModel['model'](torch.from_numpy(BindingScoreData).float()).numpy()
        else:
            score = jit_predict_BindingScore(torch.from_numpy(BindingScoreData).float(),
                                             BindingScoreModel['footprintMean'],
                                             BindingScoreModel['footprintSd'],
                                             *BindingScoreModel['weights']).numpy()

    else:
        BindingScoreData = None
        return {}

    pred_time = time.time() - start
    del BindingScoreData, multiScaleFootprints, siteFootprints
    return {"position": relativePos,
            # "region": region,
            # "sites": sites,
            "siteFilter": siteFilter,
            'BindingScore': score[..., 0], # 0, because the NN outputs a dim 1.
            "region_identifier": region_identifier,
            "time": np.array([footprint_time, stride_time, transpose_time, pred_time])}


def getRegionBindingScore_batch(
                        CountTensorPath,
                        region_identifier_chunk,
                        width, num_groups, select_group_id, seqBiass, regions,
                        dispModels,
                        BindingScoreModel, sites, tileSize, contextRadius):
    result = []
    with h5py.File(CountTensorPath, "r") as f:
        for i, region_identifier in enumerate(region_identifier_chunk):
            try:
                region_count = np.array(f[region_identifier])
            except:
                print (region_identifier, "has zero counts in the given groups")
            result.append(getRegionBindingScore(
                region_identifier,
                (region_count, num_groups, width, select_group_id),
                seqBiass[i,],
                dftopyranges(regions.iloc[i:i + 1]),
                dispModels,
                BindingScoreModel,
                sites,
                tileSize,
                contextRadius))
    return result


def getBindingScore(
            CountTensorPath,
            regions,
            groups,
            dispersionModels,
            BindingScoreModel,
            Tn5Bias,
            motifs=None,
            contextRadius=100, # well it's never ued in the R version... so... yeah
            nCores=16,
            saveDir=None,
            saveName = "chunkedBindingResults.h5",
            returnCombined=None,
            append_mode=False,  # When True, it means, the new results will be appended ore groups (mode 2)
            # / more regions (mode 1)), cannot have more reads, because that just changes the read count.
         ):

    if returnCombined is None:
        returnCombined = saveDir is None
    assert append_mode != 3, "Don't support append fragments, need to be rerun!"
    if saveDir is not None:
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

    # TODO: make sure the return combined is in the correct order.
    regions = regionparser(regions)
    groups = [str(xx) for xx in groups]
    # region_identifier is the string like chr1:0-1000 that identifies each region.
    region_identifier = df2regionidentifier(regions)
    width = regions.iloc[0]['End'] - regions.iloc[0]['Start']
    seqBias = Tn5Bias

    scales = BindingScoreModel['scales']

    # Only keep the related scales to avoid sending unnecessary stuff back and forth
    dispModels = [dispersionModels[str(scale)]
                  for scale in scales]

    with h5py.File(CountTensorPath, "r") as f:

        uniq_group, existing_counttensor_groups, group_notin_counttensor, \
            uniq_region, existing_region, region_notin_counttensor = Unify_meta_info(f,
                                                                                     addition_feats=[groups,
                                                                                                     region_identifier],
                                                                                     entries=['group', 'regions'],
                                                                                     dtypes=['str', 'str'])
        assert len(group_notin_counttensor) == 0, "observe new groups that don't have count tensor: %s..." % \
                                                  group_notin_counttensor[0]
        assert len(region_notin_counttensor) == 0, "cannot have new regions that don't have count tensor: %s..." % \
                                                   region_notin_counttensor[0]

        group_map = {g: i for i, g in enumerate(existing_counttensor_groups)}
        select_group_id = np.array([group_map[g] for g in groups])
        if np.sum(select_group_id - np.arange(len(groups))) == 0:
            select_group_id = None

    if motifs is None:
        motifPositions = None
    else:
        # OK this is yet to be implemented,
        # need to find a package for matching motifs.
        raise NotImplementedError

    if saveDir is not None:
        final_hdf5_path = os.path.join(saveDir, "%s" % saveName)
        final_hdf5 = h5py.File(final_hdf5_path, "a" if append_mode else "w")

        uniq_group, existing_group, new_group, \
            uniq_region, existing_region, new_region = Unify_meta_info(final_hdf5,
                                                                    addition_feats=[groups, region_identifier],
                                                                    entries=['group', 'regions'],
                                                                    dtypes=['str', 'str'])

        if append_mode == 1:
            # append region mode, check if there are new groups:
            assert len(new_group) == 0, "append_region mode, but observed changed cell grouping: %s ..." % \
                                        new_group[0]
            assert len(new_region) == len(uniq_region), "append mode, cannot have overlap regions: %s" % \
                                                        uniq_region[np.isin(uniq_region, existing_region)][0]
        elif append_mode == 2:
            # append group mode, check if there are new regions
            assert len(new_region) == 0, "append_group mode, but observed changed regions: %s..." % new_region[
                0]
            assert len(new_group) == len(uniq_group), "append mode, cannot have overlap groups: %s" % \
                                                      uniq_group[np.isin(uniq_group, existing_group)][0]

        if 'regions' in final_hdf5.keys():
            # delete existing groups and regions, since we'll rewrite later.
            del final_hdf5['regions'], final_hdf5['group']

        # hdf5 in pythonwants the dtype to be specific
        groups_all = list(existing_group) + list(uniq_group)
        groups_all = [str(xx) for xx in groups_all]

        final_hdf5.create_dataset('group', data=groups_all, dtype=h5py.special_dtype(vlen=str))
        regions_all = list(existing_region) + list(region_identifier)
        regions_all = sort_region_identifier(regions_all)
        final_hdf5.create_dataset('regions', data=regions_all, dtype=h5py.special_dtype(vlen=str))

    region_identifier = list(region_identifier)
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")

        small_chunk_size = min(int(math.ceil(len(regions) / nCores) + 1 ), 100)
        pool = ProcessPoolExecutor(max_workers=nCores)
        p_list = []

        for i in range(0, len(regions), small_chunk_size):
            slice_ = slice(i, i+small_chunk_size)

            p_list.append(pool.submit(
                getRegionBindingScore_batch,
                CountTensorPath,
                region_identifier[slice_],
                width,
                len(uniq_group),
                select_group_id,
                seqBias[slice_],
                regions.iloc[slice_],
                dispModels,
                BindingScoreModel,
                None, # For now... sites = None,
                10, # tile size,
                contextRadius,
            ))

        bar = trange(len(region_identifier), desc='collecting Binding prediction')


        return_dict = [0] * len(regions)
        time = 0
        for p in as_completed(p_list):
            rs = p.result()
            for r in rs:
                time += r['time']
                if saveDir is not None:
                    identifier = r['region_identifier']

                    xx = r['BindingScore']
                    if append_mode:
                        # If there's a region already, so must be append group or scale
                        if identifier in final_hdf5:
                            # shape of (# groups (old), # position)
                            old_result = np.array(final_hdf5[identifier]['BindingScore'])
                            del final_hdf5[identifier]['BindingScore']
                            if append_mode == 2:
                                # more groups? concat at the first dim
                                xx = np.concatenate([old_result, xx], axis=0)
                        # else: more regions? just create!
                    else:
                        grp = final_hdf5.create_group(identifier)
                        # grp.create_dataset('position', data=r['position'], compression="gzip")
                        # r['sites'].to_hdf(final_hdf5_path, key="/%s/sites" % identifier, format="table")

                    grp.create_dataset('BindingScore', data=xx, compression="gzip")


                if returnCombined:
                    return_dict[region_identifier.index(identifier)] = r
            bar.update(len(rs))
            bar.set_description(np.array2string(time, precision=1, separator=',',
                                                suppress_small=True))

        if saveDir is not None:
            final_hdf5.close()
        bar.close()
        pool.shutdown(wait=True)

        if returnCombined:
            return return_dict


