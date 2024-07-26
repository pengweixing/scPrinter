# adapted from bpnetlite - attribution.py

import bioframe
import numba
import numpy
import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

try:
    from scprinter.utils import DNA_one_hot, zscore2pval_torch
except ImportError:
    from ...utils import DNA_one_hot, zscore2pval_torch
from .shap_deeplift import PyTorchDeep
from .shap_expct_grad import PyTorchGradient

params = "void(int64, int64[:], int64[:], int32[:, :], int32[:,], "
params += "int32[:, :], float32[:, :, :], int32)"


# This from is from bpnetlite - attribution.py
def hypothetical_attributions(multipliers, inputs, baselines):
    """
    A function for aggregating contributions into hypothetical attributions.

    When handling categorical data, like one-hot encodings, the attributions
    returned by a method like DeepLIFT/SHAP may need to be modified slightly.
    Specifically, one needs to account for each nucleotide change actually
    being the addition of one category AND the subtraction of another category.
    Basically, once you've calculated the multipliers, you need to subtract
    out the contribution of the nucleotide actually present and then add in
    the contribution of the nucleotide you are becomming.

    These values are then averaged over all references.


    Parameters
    ----------
    multipliers: torch.tensor, shape=(n_baselines, 4, length)
        The multipliers determined by DeepLIFT

    inputs: torch.tensor, shape=(n_baselines, 4, length)
        The one-hot encoded sequence being explained, copied several times.

    baselines: torch.tensor, shape=(n_baselines, 4, length)
        The one-hot encoded baseline sequences.


    Returns
    -------
    projected_contribs: torch.tensor, shape=(1, 4, length)
        The attribution values for each nucleotide in the input.
    """
    # print (multipliers.shape, inputs.shape, baselines.shape)
    projected_contribs = torch.zeros_like(baselines[0], dtype=baselines[0].dtype)

    for i in range(inputs[0].shape[1]):
        hypothetical_input = torch.zeros_like(inputs[0], dtype=baselines[0].dtype)
        hypothetical_input[:, i] = 1.0
        hypothetical_diffs = hypothetical_input - baselines[0]
        hypothetical_contribs = hypothetical_diffs * multipliers[0]

        projected_contribs[:, i] = torch.sum(hypothetical_contribs, dim=1)

    return (projected_contribs,)


# This from is from bpnetlite - attribution.py
@numba.jit(params, nopython=False)
def _fast_shuffle(
    n_shuffles,
    chars,
    idxs,
    next_idxs,
    next_idxs_counts,
    counters,
    shuffled_sequences,
    random_state,
):
    """An internal function for fast shuffling using numba."""

    numpy.random.seed(random_state)

    for i in range(n_shuffles):
        for char in chars:
            n = next_idxs_counts[char]

            next_idxs_ = numpy.arange(n)
            next_idxs_[:-1] = numpy.random.permutation(n - 1)  # Keep last index same
            next_idxs[char, :n] = next_idxs[char, :n][next_idxs_]

        idx = 0
        shuffled_sequences[i, idxs[idx], 0] = 1
        for j in range(1, len(idxs)):
            char = idxs[idx]
            count = counters[i, char]
            idx = next_idxs[char, count]

            counters[i, char] += 1
            shuffled_sequences[i, idxs[idx], j] = 1


# This from is from bpnetlite - attribution.py
def dinucleotide_shuffle(sequence, n_shuffles=10, random_state=None):
    """Given a one-hot encoded sequence, dinucleotide shuffle it.

    This function takes in a one-hot encoded sequence (not a string) and
    returns a set of one-hot encoded sequences that are dinucleotide
    shuffled. The approach constructs a transition matrix between
    nucleotides, keeps the first and last nucleotide constant, and then
    randomly at uniform selects transitions until all nucleotides have
    been observed. This is a Eulerian path. Because each nucleotide has
    the same number of transitions into it as out of it (except for the
    first and last nucleotides) the greedy algorithm does not need to
    check at each step to make sure there is still a path.

    This function has been adapted to work on PyTorch tensors instead of
    numpy arrays. Code has been adapted from
    https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py

    Parameters
    ----------
    sequence: torch.tensor, shape=(k, -1)
        The one-hot encoded sequence. k is usually 4 for nucleotide sequences
        but can be anything in practice.

    n_shuffles: int, optional
        The number of dinucleotide shuffles to return. Default is 10.

    random_state: int or None or numpy.random.RandomState, optional
        The random seed to use to ensure determinism. If None, the
        process is not deterministic. Default is None.

    Returns
    -------
    shuffled_sequences: torch.tensor, shape=(n_shuffles, k, -1)
        The shuffled sequences.
    """

    if random_state is None:
        random_state = numpy.random.randint(0, 9999999)

    chars, idxs = torch.unique(sequence.argmax(axis=0), return_inverse=True)
    chars, idxs = chars.numpy(), idxs.numpy()

    n_chars, seq_len = sequence.shape
    next_idxs = numpy.zeros((n_chars, seq_len), dtype=numpy.int32)
    next_idxs_counts = numpy.zeros(max(chars) + 1, dtype=numpy.int32)

    for char in chars:
        next_idxs_ = numpy.where(idxs[:-1] == char)[0]
        n = len(next_idxs_)

        next_idxs[char][:n] = next_idxs_ + 1
        next_idxs_counts[char] = n

    shuffled_sequences = numpy.zeros((n_shuffles, *sequence.shape), dtype=numpy.float32)
    counters = numpy.zeros((n_shuffles, len(chars)), dtype=numpy.int32)

    _fast_shuffle(
        n_shuffles,
        chars,
        idxs,
        next_idxs,
        next_idxs_counts,
        counters,
        shuffled_sequences,
        random_state,
    )

    shuffled_sequences = torch.from_numpy(shuffled_sequences).float()
    return shuffled_sequences


def calculate_attributions(
    model,
    X,
    method="shap_hypo",
    n_shuffles=20,
    verbose=False,
    n_steps=10,
    references=None,
    amp=True,
):
    """
    Calculate attributions for a model and a set of sequences.

    Parameters
    ----------
    model: torch.nn.Module
        The model to use.
    X: torch.tensor, shape=(n, 4, seq_len)
        The input dna sequences.
    method: str, optional
        The attribution method to use. Default is 'shap_hypo'.
    n_shuffles: int, optional
        The number of background shuffles to use. Default is 20.
    verbose: bool, optional
        Whether to display a progress bar. Default is False.
    n_steps: int, optional
        The number of steps to use for integrated gradients. Default is 10.
    references:
        The reference sequences to use. If None, the references will be generated by dinucleotide shuffle
    amp: bool, optional
        Whether to use automatic mixed precision. Default is True.

    Returns
    -------

    """
    batch_size = 32 if method == "ism" else 1
    attributions = []
    dev = next(model.parameters()).device
    for i in trange(0, len(X), batch_size, disable=not verbose, dynamic_ncols=True):
        _X = X[i : i + batch_size].float()
        if method != "ism":
            if references is None:
                _references = torch.cat(
                    [dinucleotide_shuffle(xx, n_shuffles=n_shuffles).to(dev) for xx in _X],
                    dim=0,
                )
            _references = _references.type(_X.dtype)
        _X = _X.to(dev)
        if "shap" in method:
            dl = PyTorchDeep(model, _references)
            dl.amp = amp
            attr, deltas = dl.shap_values(
                _X,
                check_additivity=False,
                custom_attribution_func=(hypothetical_attributions if "hypo" in method else None),
            )
            if "hypo" not in method:
                # print (deltas)
                if torch.max(torch.abs(deltas[0])) > 1e-2:
                    print("not converging", deltas)
        elif "IxG" in method:
            model.zero_grad()
            _X.requires_grad = True
            with torch.autograd.set_grad_enabled(True):
                output = model(_X)
                attr = torch.autograd.grad(output, _X)[0]
            if "norm" in method:
                norm_factor = (attr.sum(dim=1, keepdims=True) - attr) / 3
                attr = attr - norm_factor
            if "abs" in method:
                attr = torch.abs(attr)

        elif "ism" in method:
            attr = ism(model, _X)

        elif "expct_grad" in method:
            # Haven't really benchmarked this method...
            dl = PyTorchGradient(model, _references, correction=True)
            attr = dl.shap_values(_X, nsamples=500)
        else:
            raise ValueError(f"method {method} not supported")

        if len(attr.shape) == 2:
            attr = attr[None]

        attributions.append(
            attr.cpu().detach() if type(attr) is torch.Tensor else torch.from_numpy(attr)
        )

    attributions = torch.cat(attributions)
    return attributions


@torch.no_grad()
def project_attrs(attributions, seqs, bs=64, device="cpu"):
    """
    Project hypothetical attributions onto the sequence. and collapse into 1D track

    Parameters
    ----------
    attributions: torch.Tensor, shape=(n, 4, seq_len)
        The attributions to project.
    seqs: torch.Tensor, shape=(n, 4, seq_len)
        The sequences to project onto.
    bs: int, optional
        The batch size to use. Default is 64.
    device: str, optional
        The device to use. Default is 'cpu'.

    Returns
    -------
    attributions_projected: np.array, shape=(n, seq_len)
        The projected attributions.

    """
    attributions_projected = []
    for i in range(0, len(attributions), bs):
        _attributions = attributions[i : i + bs].to(device)
        _seqs = seqs[i : i + bs].to(device)
        _attributions_projected = (_attributions * _seqs).sum(dim=1)
        attributions_projected.append(_attributions_projected.detach().cpu().numpy())
    attributions_projected = np.concatenate(attributions_projected)
    return attributions_projected


def attribution_to_bigwig(
    attributions,
    regions,
    chrom_size,
    res=1,
    mode="average",
    output="attribution.bigwig",
    verbose=True,
):
    """
    A function that takes projected attrs and output a bigwig file. This function takes into consideration when regions are overlapped
    with each other. and will be averaged or summed or maxed based on the mode.

    Parameters
    ----------
    attributions
    seqs
    regions
    mode
    output

    Returns
    -------

    """
    regions = regions.copy()
    regions.columns = ["chrom", "start", "end"] + list(regions.columns[3:])

    bw = pyBigWig.open(output, "w")

    regions = bioframe.cluster(regions)  # group regions that are overlapped
    regions["fake_id"] = np.arange(len(regions))
    regions = regions.sort_values(by=["chrom", "start", "end"])
    cluster_stats = regions.drop_duplicates("cluster")
    cluster_stats = cluster_stats.sort_values(by=["chrom", "start", "end"])
    order_of_cluster = np.array(cluster_stats["cluster"])
    aa = regions.groupby("cluster")

    header = []
    chrom_order = cluster_stats["chrom"].unique()
    for chrom in chrom_order:
        header.append((chrom, int(chrom_size[chrom])))
    bw.addHeader(header, maxZooms=10)

    stuff_to_add = {"chroms": None, "points": [], "values": []}

    for cluster in tqdm(order_of_cluster, dynamic_ncols=True, disable=not verbose):
        region_temp = aa.get_group(cluster)
        chroms, starts, ends = (
            np.array(region_temp["chrom"]),
            region_temp["start"],
            region_temp["end"],
        )
        mask = np.array(region_temp["fake_id"])
        attributions_chrom = attributions[mask]

        if str(chroms[0]) != stuff_to_add["chroms"] and stuff_to_add["chroms"] is not None:
            chrom = stuff_to_add["chroms"]
            points = np.concatenate(stuff_to_add["points"])
            values = np.concatenate(stuff_to_add["values"])

            bw.addEntries(chrom, points, values=values, span=res)
            stuff_to_add = {"chroms": None, "points": [], "values": []}

        if len(mask) == 1:
            # No need to deal with overlapping things
            chrom = np.array(chroms)[0]
            points = np.arange(attributions_chrom[0].shape[-1]) * res + np.array(starts)[0]
            stuff_to_add["chroms"] = str(chrom)
            stuff_to_add["points"].append(points)
            stuff_to_add["values"].append(attributions_chrom[0].astype("float"))
            continue

        # There're some overlapping thing, create a matrix of (# overlapping region, # maximum reigon that covers all)
        start_point = np.min(starts)
        end_point = np.max(ends)
        size = (end_point - start_point) // res
        importance_track = np.full((len(region_temp), size), np.nan)
        ct = 0
        for start, end, attribution in zip(starts, ends, attributions_chrom):
            if end - start != attribution.shape[0] * res:
                print("shape incompatible", start, end, attribution.shape)
            importance_track[ct, (start - start_point) // res : (end - start_point) // res] = (
                attribution
            )
            ct += 1

        if mode == "average":
            importance_track = np.nanmean(importance_track, axis=0)
        elif mode == "max":
            importance_track = np.nanmax(importance_track, axis=0)
        elif mode == "min":
            importance_track = np.nanmin(importance_track, axis=0)
        elif mode == "sum":
            importance_track = np.nansum(importance_track, axis=0)

        indx = ~np.isnan(importance_track)
        chrom = np.array(chroms)[0]
        points = np.array(np.where(indx)[0]) * res + start_point

        stuff_to_add["chroms"] = str(chrom)
        stuff_to_add["points"].append(points)
        stuff_to_add["values"].append(importance_track[indx])

    chrom = stuff_to_add["chroms"]
    points = np.concatenate(stuff_to_add["points"])
    values = np.concatenate(stuff_to_add["values"])

    bw.addEntries(chrom, points, values=values, span=res)

    bw.close()


def count_delta(alt, ref):
    alt = alt[1].detach().cpu().numpy()
    ref = ref[1].detach().cpu().numpy()
    return float(alt - ref)


def footprint_delta(alt, ref):
    alt = alt[0].reshape((1, -1))
    ref = ref[0].reshape((1, -1))
    alt = alt - torch.mean(alt, dim=-1, keepdims=True)
    ref = ref - torch.mean(ref, dim=-1, keepdims=True)
    alt = zscore2pval_torch(alt)
    ref = zscore2pval_torch(ref)
    alt = F.relu(alt - 0.31)
    ref = F.relu(ref - 0.31)
    return float(alt.sum() - ref.sum())


def tf_footprint_delta(alt, ref):
    alt = alt[0][:, :30].reshape((1, -1))
    ref = ref[0][:, :30].reshape((1, -1))
    alt = alt - torch.mean(alt, dim=-1, keepdims=True)
    ref = ref - torch.mean(ref, dim=-1, keepdims=True)
    alt = zscore2pval_torch(alt)
    ref = zscore2pval_torch(ref)
    alt = F.relu(alt - 0.31)
    ref = F.relu(ref - 0.31)
    return float(alt.sum() - ref.sum())


@torch.no_grad()
def vcf_attribution(vcf_path, seq_len, fasta, model, delta_func=None):
    ref_seq = pyfaidx.Fasta(fasta)
    vcf = pd.read_csv(vcf_path, sep="\t", header=None)
    vcf.columns = ["chrom", "pos", "name", "REF", "ALT", "sth1", "sth2"]
    kept = vcf["REF"].isin(["A", "C", "T", "G"]) & vcf["ALT"].isin(["A", "C", "T", "G"])
    vcf = vcf.loc[kept]
    vcf["start"] = vcf["pos"] - 1 - seq_len // 2
    vcf["end"] = vcf["pos"] - 1 + seq_len // 2
    center = seq_len // 2
    dev = next(model.parameters()).device
    deltas = []
    for i in trange(len(vcf)):
        chrom, start, end = vcf.iloc[i][["chrom", "start", "end"]]
        seq = ref_seq[chrom][start:end].seq.upper()
        if seq[center] != vcf.iloc[i]["REF"]:
            print(seq[center - 1 : center + 2], vcf.iloc[i]["REF"])
            raise ValueError
        # else:
        # print(seq[center - 1:center + 1], vcf.iloc[i]['REF'])

        _X = DNA_one_hot(seq)
        _X = _X[None]
        _ref = model(_X.float().to(dev))
        seq = list(seq)
        seq[center] = vcf.iloc[i]["ALT"]
        seq = "".join(seq)
        _X = DNA_one_hot(seq)
        _X = _X[None]
        _alt = model(_X.float().to(dev))
        delta = delta_func(_alt, _ref)
        deltas.append(delta)
    vcf["deltas"] = deltas
    return vcf


@torch.no_grad()
def ism(model, X_0, args=None, batch_size=128, verbose=False):
    """In-silico mutagenesis saliency scores.

    This function will perform in-silico mutagenesis in a naive manner, i.e.,
    where each input sequence has a single mutation in it and the entirety
    of the sequence is run through the given model. It returns the ISM score,
    which is a vector of the L2 difference between the reference sequence
    and the perturbed sequences with one value for each output of the model.

    Parameters
    ----------
    model: torch.nn.Module
        The model to use.

    X_0: torch.tensor, shape=(batch_size, 4, seq_len)
        The one-hot encoded sequence to calculate saliency for.

    args: tuple or None, optional
        Additional arguments to pass into the forward function. If None,
        pass nothing additional in. Default is None.

    batch_size: int, optional
        The size of the batches.

    verbose: bool, optional
        Whether to display a progress bar as positions are being mutated. One
        display bar will be printed for each sequence being analyzed. Default
        is False.

    Returns
    -------
    X_ism: torch.tensor, shape=(batch_size, 4, seq_len)
        The saliency score for each perturbation.
    """
    # print (X_0.shape)
    n_seqs, n_choices, seq_len = X_0.shape
    X_idxs = X_0.argmax(axis=1)

    n = seq_len * (n_choices - 1)
    X = torch.tile(X_0, (n, 1, 1))
    X = X.reshape(n, n_seqs, n_choices, seq_len).permute(1, 0, 2, 3)

    for i in range(n_seqs):
        for k in range(1, n_choices):
            idx = np.arange(seq_len) * (n_choices - 1) + (k - 1)

            X[i, idx, X_idxs[i], np.arange(seq_len)] = 0
            X[i, idx, (X_idxs[i] + k) % n_choices, np.arange(seq_len)] = 1

    model = model.eval()

    if args is None:
        reference = model(X_0).unsqueeze(1)
    else:
        reference = model(X_0, *args).unsqueeze(1)

    starts = np.arange(0, X.shape[1], batch_size)
    isms = []
    for i in range(n_seqs):
        ism = []
        for start in tqdm(starts, disable=not verbose):
            X_ = X[i, start : start + batch_size].cuda()

            if args is None:
                y = model(X_)
            else:
                args_ = tuple(a[i : i + 1] for a in args)
                y = model(X_, *args_)

            ism.append(y - reference[i])

        ism = torch.cat(ism)
        if len(ism.shape) > 1:
            ism = ism.sum(dim=list(range(1, len(ism.shape))))
        isms.append(ism)

    isms = torch.stack(isms)
    isms = isms.reshape(n_seqs, seq_len, n_choices - 1)

    j_idxs = torch.arange(n_seqs * seq_len)
    X_ism = torch.zeros(n_seqs * seq_len, n_choices, device="cuda")
    for i in range(1, n_choices):
        i_idxs = (X_idxs.flatten() + i) % n_choices
        X_ism[j_idxs, i_idxs] = isms[:, :, i - 1].flatten()

    X_ism = X_ism.reshape(n_seqs, seq_len, n_choices).permute(0, 2, 1)
    X_ism = X_ism - X_ism.mean(dim=1, keepdims=True)
    return X_ism
