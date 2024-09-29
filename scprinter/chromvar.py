import logging
from typing import Literal, Union

import anndata
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sparse
from anndata import AnnData
from pynndescent import NNDescent
from scipy.sparse import csr_matrix as scipy_csr_matrix
from tqdm.auto import tqdm, trange

from scprinter.utils import get_peak_bias


def sample_bg_peaks(
    adata,
    genome,
    method: Literal["nndescent", "chromvar"] = "nndescent",
    bg_set=None,
    niterations=50,
    w=0.1,
    bs=50,
    n_jobs=1,
    gc_bias=True,
):
    """
    This function samples background peaks for chromVAR analysis in single-cell ATAC-seq data.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell ATAC-seq data.
    genome : str or scp.genome.Genome
        scp.genome.Genome object or string specifying the genome.
    method : str, optional
        Method to use for sampling background peaks. Can be either "nndescent" or "chromvar".
        Defaults to "nndescent".
    bg_set : array-like, optional
        Custom background peak set to use. The background peaks would only be selected from
        this set of peaks, instead of all peaks present in the dataset. Only applicable when
        method is "nndescent". Defaults to None.
    niterations : int, optional
        Number of nearest neighbors to sample for each peak. Defaults to 50.
    w : float, optional
        Width parameter for Gaussian kernel density estimation. Defaults to 0.1. Only used in
        "chromvar" mode.
    bs : int, optional
        Bin size for creating bins and sampling background peaks. Defaults to 50. Only used in
        "chromvar" mode.
    n_jobs : int, optional
        Number of jobs to run in parallel for nearest neighbor calculation. Defaults to 1.
    gc_bias : bool, optional
        Whether to consider GC bias in sampling background peaks. Defaults to True.

    Returns
    -------
    array-like
        Array containing indices of sampled background peaks for each peak.
    """
    if gc_bias:
        get_peak_bias(adata, genome)
    assert method in ["nndescent", "chromvar"], "Method not supported"
    reads_per_peak = adata.X.sum(axis=0)
    assert np.min(reads_per_peak) > 0, "Some peaks have no reads"
    reads_per_peak = np.log10(reads_per_peak)
    reads_per_peak = np.array(reads_per_peak).reshape((-1))
    if gc_bias:
        mat = np.array([reads_per_peak, adata.var["gc_content"].values])
        chol_cov_mat = np.linalg.cholesky(np.cov(mat))
        trans_norm_mat = scipy.linalg.solve_triangular(
            a=chol_cov_mat, b=mat, lower=True
        ).transpose()
    else:
        mat = np.array([reads_per_peak])
        trans_norm_mat = mat.transpose()

    print("Sampling nearest neighbors")
    if bg_set is not None:
        assert (
            method == "nndescent"
        ), "Only nndescent method is supported for custom background peaks"
    if method == "nndescent":
        index = NNDescent(
            trans_norm_mat[bg_set if bg_set is not None else slice(None)],
            metric="euclidean",
            n_neighbors=niterations + 1,
            n_jobs=n_jobs,
        )
        knn_idx, _ = index.query(trans_norm_mat, niterations + 1)
        knn_idx = knn_idx[:, 1:]
        if bg_set is not None:
            knn_idx = bg_set[knn_idx.reshape((-1))].reshape((-1, niterations))
        adata.varm["bg_peaks"] = knn_idx
    elif method == "chromvar":
        knn_idx = create_bins_and_sample_background(
            trans_norm_mat, bs=bs, w=w, niterations=niterations
        )
        adata.varm["bg_peaks"] = knn_idx

    return knn_idx


sample_bg_peaks2 = sample_bg_peaks


def create_bins_and_sample_background(trans_norm_mat, bs, w, niterations):
    """
    Translated from the chromVAR R package.
    Parameters
    ----------
    trans_norm_mat
    bs
    w
    niterations

    Returns
    -------

    """
    # Create bins
    bins1 = np.linspace(np.min(trans_norm_mat[:, 0]), np.max(trans_norm_mat[:, 0]), bs)
    bins2 = np.linspace(np.min(trans_norm_mat[:, 1]), np.max(trans_norm_mat[:, 1]), bs)

    # Create bin_data
    bin_data = np.array(np.meshgrid(bins1, bins2)).T.reshape(-1, 2)

    # Calculate Euclidean distances
    bin_dist = scipy.spatial.distance.cdist(bin_data, bin_data, "euclidean")

    # Calculate probabilities
    bin_p = scipy.stats.norm.pdf(bin_dist, 0, w)
    # Find nearest bin membership for each point in trans_norm_mat
    print("NNDescent", bin_data.shape)
    index = NNDescent(bin_data, metric="euclidean", n_neighbors=1, n_jobs=1)
    indices, _ = index.query(trans_norm_mat, 1)
    # distance = scipy.spatial.distance.cdist(trans_norm_mat, bin_data)
    # indices = np.argmin(distance, axis=1)
    bin_membership = indices.flatten()
    # Calculate bin density
    unique, counts = np.unique(bin_membership, return_counts=True)
    bin_density = np.zeros(bs**2)
    bin_density[unique] = counts

    # Sample background peaks
    # This assumes bg_sample_helper is defined as per previous instruction
    background_peaks = bg_sample_helper(bin_membership, bin_p, bin_density, niterations)

    return background_peaks


def bg_sample_helper(bin_membership, bin_p, bin_density, niterations):
    n = len(bin_membership)
    out = np.zeros((n, niterations), dtype=int)

    for i in trange(len(bin_density), desc="Sampling background peaks"):
        ix = np.where(bin_membership == i)[0]
        if len(ix) == 0:  # Skip if no members in bin
            continue
        p_tmp = bin_p[i,]
        p = (p_tmp / bin_density)[bin_membership]
        p /= p.sum()
        # Sampling with replacement according to probabilities
        sampled_indices = np.random.choice(
            np.arange(len(p)), size=niterations * len(ix), replace=True, p=p
        )
        out[ix, :] = sampled_indices.reshape((len(ix), niterations))

    return out


def scipy_to_cupy_sparse(sparse_matrix):
    """
    A function that converts a SciPy sparse matrix to a CuPy sparse matrix. Only supports CSR matrices now.
    Parameters
    ----------
    sparse_matrix

    Returns
    -------

    """
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix

    if not isinstance(sparse_matrix, scipy_csr_matrix):
        raise ValueError("Input matrix must be a SciPy CSR matrix")

    # Get the CSR components of the SciPy sparse matrix
    data = sparse_matrix.data.astype("float")
    indices = sparse_matrix.indices
    indptr = sparse_matrix.indptr
    shape = sparse_matrix.shape

    # Convert the components to CuPy arrays
    data_cp = cp.array(data)
    indices_cp = cp.array(indices)
    indptr_cp = cp.array(indptr)

    # Create a CuPy CSR matrix with these components
    cupy_sparse_matrix = cupy_csr_matrix((data_cp, indices_cp, indptr_cp), shape=shape)
    return cupy_sparse_matrix


def compute_deviations(adata, chunk_size: int = 10000, device="cuda"):
    """
    Computes the deviation of motif matches from the background for each cell.

    Parameters
    ----------
    adata : AnnData
        The input AnnData object containing the count matrix, background peaks, and motif match information.
    chunk_size : int, optional
        The size of chunks to process the data in. Default is 10000. It is recommended to set this
        such that there's no GPU memory overflow. Although our implementation would tolerate that,
        it will be much slower if there's overflow.
    device : str, optional
        The device to use for computation. Can be either "cuda" for GPU or "cpu" for CPU. Default is "cuda".

    Returns
    -------
    dev : AnnData
        The AnnData object containing the deviation values for each cell and motif match.
    """

    assert (
        "bg_peaks" in adata.varm_keys()
    ), "Cannot find background peaks in the input object, please first run get_bg_peaks!"
    if device == "cuda":
        import cupy as backend
    else:
        import numpy as backend

    print("Computing expectation reads per cell and peak...")
    expectation_var = backend.asarray(adata.X.sum(0), dtype=backend.float32).reshape(
        (1, adata.X.shape[1])
    )
    expectation_var /= expectation_var.sum()
    expectation_obs = np.asarray(adata.X.sum(1), dtype=np.float32).reshape((adata.X.shape[0], 1))
    motif_match = backend.asarray(adata.varm["motif_match"], dtype=backend.float32)

    obs_dev = np.zeros((adata.n_obs, motif_match.shape[1]), dtype=np.float32)
    n_bg_peaks = adata.varm["bg_peaks"].shape[1]
    # bg_dev = np.zeros((n_bg_peaks, adata.n_obs, motif_match.shape[1]), dtype=np.float32)
    mean_bg_dev = np.zeros_like(obs_dev)
    std_bg_dev = np.zeros_like(obs_dev)

    for start in tqdm(range(0, adata.n_obs, chunk_size), desc="Processing chunks"):
        end = min(start + chunk_size, adata.n_obs)
        temp_adata = adata[start:end].copy()
        X_chunk = temp_adata.X
        expectation_obs_chunk = backend.asarray(expectation_obs[start:end])
        if sparse.isspmatrix(X_chunk):
            if device == "cuda":
                X_chunk = scipy_to_cupy_sparse(X_chunk)
            else:
                X_chunk = X_chunk.tocsr()
        else:
            X_chunk = backend.array(X_chunk)
        res = _compute_deviations(
            motif_match,
            X_chunk,
            expectation_obs_chunk,
            expectation_var,
            device=device,
        )
        obs_dev[start:end, :] = res.get() if device == "cuda" else res
        bg_dev_chunk = np.zeros((n_bg_peaks, end - start, motif_match.shape[1]), dtype=np.float32)
        for i in trange(n_bg_peaks, desc="Processing background peaks"):
            bg_peak_idx = backend.array(adata.varm["bg_peaks"][:, i]).flatten()
            bg_motif_match = motif_match[bg_peak_idx, :]
            res = _compute_deviations(
                bg_motif_match,
                X_chunk,
                expectation_obs_chunk,
                expectation_var,
                device=device,
            )
            bg_dev_chunk[i, :, :] = res.get() if device == "cuda" else res
        mean_bg_dev[start:end, :] = np.mean(bg_dev_chunk, axis=0)
        std_bg_dev[start:end, :] = np.std(bg_dev_chunk, axis=0)
        del temp_adata, X_chunk

    # mean_bg_dev = np.mean(bg_dev, axis=0)
    # std_bg_dev = np.std(bg_dev, axis=0)
    dev = (obs_dev - mean_bg_dev) / std_bg_dev
    dev = np.nan_to_num(dev, nan=0.0)

    dev = AnnData(
        dev, dtype="float32", obs=adata.obs.copy()
    )  # Convert back to CPU for AnnData compatibility
    dev.var_names = adata.uns["motif_name"]
    return dev


def _compute_deviations(motif_match, count, expectation_obs, expectation_var, device):
    if device == "cuda":
        import cupy as backend
    else:
        import numpy as backend

    observed = count.dot(motif_match)
    expected = expectation_obs.dot(expectation_var.dot(motif_match))
    out = backend.zeros_like(expected)
    backend.divide(observed - expected, expected, out=out)
    out[expected == 0] = 0
    return out


def bag_deviations(adata=None, ranked_df=None, cor=0.7, motif_corr_matrix=None):
    """
    This function performs a bagging operation on transcription factors (TFs) based on their correlation with each other.
    It selects a representative TF (sentinel TF) for each group of TFs that have a correlation coefficient greater than or equal to the specified threshold.

    Parameters
    ----------
    adata : AnnData, optional
        An AnnData object containing the count matrix and TF information. If provided, TF variability will be computed from this object.
    ranked_df : DataFrame, optional
        A DataFrame containing TF names and their variability scores. If provided, TF variability will be computed from this object.
    cor : float, optional
        The correlation coefficient threshold. TFs with a correlation coefficient greater than or equal to this threshold will be grouped together.
    motif_corr_matrix : str, optional
        The path to the motif correlation matrix file. The common options are `scp.datasets.FigR_motifs_bagging_mouse` and `scp.datasets.FigR_motifs_bagging_human`.
    use_name : bool, optional
        A flag indicating whether to use TF names from the correlation matrix file. If True, TF names will be extracted from the correlation matrix file.

    Returns
    -------
    DataFrame
        If an AnnData object is provided, a DataFrame containing the selected sentinel TFs and their corresponding TF groups.
    list
        If a DataFrame object is provided, a list containing the selected sentinel TFs.
    list
        If a DataFrame object is provided, a list containing the TF groups.
    """

    assert motif_corr_matrix is not None, "Motif correlation matrix must be provided"
    assert adata is not None or ranked_df is not None, "Either adata or ranking_df must be provided"

    # Compute variability and get transcription factors (TFs)
    if adata is not None:
        x = adata.X
        if type(adata.X) not in [np.ndarray]:
            x = x.get()
        vb = np.nanstd(x, axis=0)
        vb = pd.DataFrame({"variability": vb}, index=adata.var.index)
        vb = vb.sort_values(by=["variability"], ascending=False)
        TFnames = vb.index.tolist()
    else:
        TFnames = ranked_df.index.tolist()
    TFnames_to_rank = {tf: i for i, tf in enumerate(TFnames)}
    # Import correlation based on PWMs for the organism
    if type(motif_corr_matrix) is pd.DataFrame:
        cormat = motif_corr_matrix
    else:
        cormat = pd.read_csv(
            motif_corr_matrix, sep="\t"
        )  # Assuming the RDS file contains one object

    # Historical code, kept for future references
    # if use_name:
    #     tf1 = [xx.split("_")[2] for xx in cormat["TF1"]]
    #     tf2 = [xx.split("_")[2] for xx in cormat["TF2"]]
    #     cormat["TF1"] = tf1
    #     cormat["TF2"] = tf2

    assert set(TFnames).issubset(
        set(cormat["TF1"]).union(set(cormat["TF2"]))
    ), "All TF names must be in the correlation matrix"
    cormat = cormat[(cormat["TF1"].isin(TFnames)) & (cormat["TF2"].isin(TFnames))]
    i = 1
    TFgroups = []
    while len(TFnames) != 0:
        tfcur = TFnames[0]
        boo = ((cormat["TF1"] == tfcur) | (cormat["TF2"] == tfcur)) & (cormat["Pearson"] >= cor)
        hits = cormat[boo]
        tfhits = list(set(list(np.unique(hits[["TF1", "TF2"]])) + [tfcur]))

        # Update lists
        TFnames = [tf for tf in TFnames if tf not in tfhits]
        TFgroups.append(tfhits)
        cormat = cormat[cormat["TF1"].isin(TFnames) & cormat["TF2"].isin(TFnames)]
        i += 1

    sentinalTFs = []
    for group in TFgroups:
        ranks = [TFnames_to_rank[tf] if tf in TFnames_to_rank else 1e9 for tf in group]
        sentinalTFs.append(group[np.argmin(ranks)])
    if adata is not None:
        objReduced = adata.var.loc[sentinalTFs]
        objReduced["groups"] = TFgroups
        return objReduced
    else:
        return sentinalTFs, TFgroups
