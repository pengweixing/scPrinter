import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
from .footprint import *
from .io import get_global_bindingscore_model, get_global_disp_models
from .utils import *


# torch jit version
# Currently with fixed structure, the gpu version already assumes no structure, should change it later
@torch.jit.script
def jit_predict_BindingScore(
    data,
    footprintMean,
    footprintSd,
    weights_0,
    weights_1,
    weights_2,
    weights_3,
    weights_4,
    weights_5,
):

    data = (data - footprintMean) / footprintSd

    if data.shape[-1] != weights_0.shape[0]:
        x = F.relu(F.linear(data, weights_0[:-1].T, weights_1 + weights_0[-1]))
    else:
        x = F.relu(F.linear(data, weights_0.T, weights_1))
    x = F.relu(F.linear(x, weights_2.T, weights_3))
    x = torch.sigmoid(F.linear(x, weights_4.T, weights_5))
    return x


@torch.no_grad()
def bindingscore(
    regionATAC,  # Position-by-pseudobulk matrix of ATAC data for the current region
    Tn5Bias,  # Numeric vector of predicted Tn5 bias for the current region
    dispModels,  # Background dispersion model for center-vs-(center + flank) ratio insertion ratio
    BindingScoreModel,  # Model for predicting TF binding at any motif site
    contextRadius=100,  # Local radius of model input (in bp)
):
    """
    This function calculates TF/Nucleosome binding scores for a given region using a pre-trained model
    based on footprints.
    This is a wrapper function to just get the binding scores without parallelization, mainly for visualization.
    And it doesn't consider motif matching scores etc.

    Parameters:
    regionATAC (numpy.ndarray): Position-by-pseudobulk matrix of ATAC data for the current region.
    Tn5Bias (numpy.ndarray): Numeric vector of predicted Tn5 bias for the current region.
    dispModels (dict): Background dispersion model for center-vs-(center + flank) ratio insertion ratio.
    BindingScoreModel (dict): Model for predicting TF binding at any motif site.
    contextRadius (int, optional): Local radius of model input (in bp). Default is 100.

    Returns:
    numpy.ndarray: Predicted TF binding scores for the given region.
    """
    width = len(Tn5Bias)
    scales = BindingScoreModel["scales"]  # scales should be coded in the BindingScoreModel
    multiScaleFootprints = None

    start = time.time()
    regionATAC = regionATAC

    for scale_index, scale in enumerate(scales):
        # I changed the code in regionFootprintscore to make it consistent
        footprintScores, _ = regionFootprintScore(
            regionATAC, Tn5Bias, dispModels[str(scale)], scale, scale, None
        )
        if multiScaleFootprints is None:
            # shape of (group, scale, position)
            multiScaleFootprints = np.zeros(
                (footprintScores.shape[0], len(scales), footprintScores.shape[1]),
                dtype="float32",
            )
        multiScaleFootprints[:, scale_index, :] = footprintScores

    footprint_time = time.time() - start
    # Only keep sites with distance to CRE edge >= contextRadius
    start_site = np.arange(0, width - contextRadius * 2)
    slice_width = 2 * contextRadius + 1
    # This is the stride trick
    # shape of, (group, scale, sites, 2 * contextRadius + 1)
    siteFootprints = strided_lastaxis(multiScaleFootprints, slice_width)[:, :, start_site]
    stride_time = time.time() - start
    start = time.time()

    # shape of (group, sites, scale, 2 * context Radius + 1)
    # then reshaped to (group, sites, scale * (2 * context Radius + 1))
    BindingScoreData = np.transpose(siteFootprints, axes=[0, 2, 1, 3]).reshape(
        (siteFootprints.shape[0], siteFootprints.shape[2], -1)
    )

    transpose_time = time.time() - start
    start = time.time()

    if "model" in BindingScoreModel:
        # For some of the models I make "model" in BindingScoreModel, so we can assume no structures.
        score = BindingScoreModel["model"](torch.from_numpy(BindingScoreData).float()).numpy()
    else:
        score = jit_predict_BindingScore(
            torch.from_numpy(BindingScoreData).float(),
            BindingScoreModel["footprintMean"],
            BindingScoreModel["footprintSd"],
            *BindingScoreModel["weights"],
        ).numpy()

    pred_time = time.time() - start
    del BindingScoreData, multiScaleFootprints, siteFootprints
    return score[..., 0]  # 0, because the NN outputs a dim 1.


# Get predicted TF binding scores for a specific region
def getRegionBindingScore(
    region_identifier,  # an identifier for things that finished
    regionATAC,  # Position-by-pseudobulk matrix of ATAC data for the current region
    Tn5Bias,  # Numeric vector of predicted Tn5 bias for the current region
    region,  # pyRanges object for the current region
    dispModels,  # Background dispersion model for center-vs-(center + flank) ratio insertion ratio
    BindingScoreModel,  # Model for predicting TF binding at any motif site
    sites=None,  # pyRanges object. Genomic ranges of motif matches in the current region
    # Must have $score attribute which indicates how well the motif is matched (between 0 and 1)
    # If NULL, the region will be divided into equally sized tiles for TF binding scoring
    tileSize=10,  # Size of tiles if sites is NULL
    contextRadius=100,  # Local radius of model input (in bp),
    strand="*",
):
    """
    Calculates TF/Nucleosome binding scores for a given region using a pre-trained model based on footprints.
    This is a wrapper function to get the binding score with tile window for a given region.

    Parameters:
    region_identifier (str): An identifier for things that finished.
    regionATAC (numpy.ndarray): Position-by-pseudobulk matrix of ATAC data for the current region.
    Tn5Bias (numpy.ndarray): Numeric vector of predicted Tn5 bias for the current region.
    region (pyranges.PyRanges): pyRanges object for the current region.
    dispModels (dict): Background dispersion model for center-vs-(center + flank) ratio insertion ratio.
    BindingScoreModel (dict): Model for predicting TF binding at any motif site.
    sites (pyranges.PyRanges, optional): pyRanges object. Genomic ranges of motif matches in the current region. Defaults to None.
    tileSize (int, optional): Size of tiles if sites is None. Defaults to 10.
    contextRadius (int, optional): Local radius of model input (in bp). Defaults to 100.
    strand (str, optional): Strand to consider. Defaults to "*".

    Returns:
    dict: A dictionary containing the following keys:
       position (numpy.ndarray): Positions of TF sites relative to the start of the CRE region.
       siteFilter (numpy.ndarray): Boolean array indicating which sites passed the filter.
       BindingScore (numpy.ndarray): Predicted TF binding scores for each pseudobulk.
       region_identifier (str): Identifier for the region.
       time (numpy.ndarray): Array containing the time taken for different steps in the function.
    """

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
        sites["End"].iloc[-1] += 1
        sites["Score"] = 1
        sites["TF"] = ""
        ## Yan, could you confirm if it's OK to do this?
        skip_site_check = False
    if len(sites) == 0:
        return {}
    if "Strand" not in sites:
        sites["Strand"] = "*"
        skip_site_check = True
    else:
        skip_site_check = False
    if strand != "*":
        skip_site_check = True

    width = len(Tn5Bias)
    scales = BindingScoreModel["scales"]
    multiScaleFootprints = None

    start = time.time()
    if type(regionATAC) is tuple:
        regionATAC = getRegionATAC(*regionATAC)
    else:
        regionATAC = regionATAC

    for scale_index, scale in enumerate(scales):
        # I changed the code in regionFootprintscore to make it consistent
        footprintScores, _ = regionFootprintScore(
            regionATAC, Tn5Bias, dispModels[scale_index], scale, scale, None
        )
        if multiScaleFootprints is None:
            # shape of (group, scale, position)
            multiScaleFootprints = np.zeros(
                (footprintScores.shape[0], len(scales), footprintScores.shape[1]),
                dtype="float32",
            )
        multiScaleFootprints[:, scale_index, :] = footprintScores

    footprint_time = time.time() - start

    # Calculate positions of TF sites relative to the start of the CRE region
    relativePos = (
        np.array(sites["Start"] + sites["End"]) * 0.5 - int(region.df["Start"].iloc[0])
    ).astype("int")
    # Only keep sites with distance to CRE edge >= contextRadius
    siteFilter = (relativePos > contextRadius) & (relativePos <= (width - contextRadius))
    if np.sum(siteFilter) == 0:
        print(sites, region, relativePos, contextRadius, width - contextRadius)
        print("no sites in the region")
    sites = sites.iloc[siteFilter]
    relativePos = relativePos[siteFilter]

    # Go through each site and calculate predicted TF binding score for each pseudobulk
    score = np.array(sites["Score"])
    score = score[None, :, None]
    start = time.time()

    if len(sites) > 0:
        start_site = relativePos - contextRadius - 1
        slice_width = 2 * contextRadius + 1
        # This is the stride trick
        # shape of, (group, scale, sites, 2 * contextRadius + 1)
        siteFootprints = strided_lastaxis(multiScaleFootprints, slice_width)[:, :, start_site]

        if not skip_site_check:
            mask = np.isin(sites["Strand"], ["-", "-1", -1])
            # print ("neg strand", np.sum(mask))
            siteFootprints[:, :, mask] = siteFootprints[:, :, mask][..., ::-1]
        if strand == "-":
            siteFootprints = siteFootprints[..., ::-1]

        stride_time = time.time() - start
        start = time.time()

        # shape of (group, sites, scale, 2 * context Radius + 1)
        # then reshaped to (group, sites, scale * (2 * context Radius + 1))
        BindingScoreData = np.transpose(siteFootprints, axes=[0, 2, 1, 3]).reshape(
            (siteFootprints.shape[0], siteFootprints.shape[2], -1)
        )
        # BindingScoreData = np.concatenate([BindingScoreData,
        #                                    score], axis=-1)

        transpose_time = time.time() - start
        start = time.time()

        # score = torch_predict_BindingScore(BindingScoreData, BindingScoreModel)
        if "model" in BindingScoreModel:
            score = BindingScoreModel["model"](torch.from_numpy(BindingScoreData).float()).numpy()
        else:
            score = jit_predict_BindingScore(
                torch.from_numpy(BindingScoreData).float(),
                BindingScoreModel["footprintMean"],
                BindingScoreModel["footprintSd"],
                *BindingScoreModel["weights"],
            ).numpy()

    else:
        BindingScoreData = None
        return {}

    pred_time = time.time() - start
    del BindingScoreData, multiScaleFootprints, siteFootprints
    return {
        "position": relativePos,
        # "region": region,
        # "sites": sites,
        "siteFilter": siteFilter,
        "BindingScore": score[..., 0],  # 0, because the NN outputs a dim 1.
        "region_identifier": region_identifier,
        "time": np.array([footprint_time, stride_time, transpose_time, pred_time]),
    }


def _bigwig_bindingscore(insertion, bias, model_key, chrom, s, e, pad=0, extra=None):
    b = np.array(bias.values(chrom, s - pad, e + pad))
    b[np.isnan(b)] = 0.0

    a = np.array(insertion.values(chrom, s - pad, e + pad))
    a[np.isnan(a)] = 0.0

    v = bindingscore(
        a[None],
        b,
        get_global_disp_models(),
        get_global_bindingscore_model()[model_key],
        contextRadius=100,
    )[0]

    if extra is not None:
        return v, extra
    return v
