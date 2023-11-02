# adapted from bpnetlite - attribution.py

import time
import numpy
import numba
import torch
import pandas
import logomaker
import pyBigWig
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from tqdm.auto import trange
import numpy as np
from tqdm.auto import tqdm, trange
import bioframe
from captum.attr import DeepLiftShap


class ProfileWrapper(torch.nn.Module):
    """A wrapper class that returns transformed profiles.

    This class takes in a trained model and returns the weighted softmaxed
    outputs of the first dimension. Specifically, it takes the predicted
    "logits" and takes the dot product between them and the softmaxed versions
    of those logits. This is for convenience when using captum to calculate
    attribution scores.

    Parameters
    ----------
    model: torch.nn.Module
        A torch model to be wrapped.
    """

    def __init__(self, model):
        super(ProfileWrapper, self).__init__()
        self.model = model

    def forward(self, X, **kwargs):
        logits = self.model(X, **kwargs)[0]
        logits = logits.reshape(X.shape[0], -1)
        logits = logits - torch.mean(logits, dim=-1, keepdims=True)

        with torch.no_grad():
            l = torch.clone(logits).detach()
            y = torch.exp(l - torch.logsumexp(l, dim=-1, keepdims=True))

        return (logits * y).sum(axis=-1, keepdims=True)


class CountWrapper(torch.nn.Module):
    """A wrapper class that only returns the predicted counts.

    This class takes in a trained model and returns only the second output.
    For BPNet models, this means that it is only returning the count
    predictions. This is for convenience when using captum to calculate
    attribution scores.

    Parameters
    ----------
    model: torch.nn.Module
        A torch model to be wrapped.
    """

    def __init__(self, model):
        super(CountWrapper, self).__init__()
        self.model = model

    def forward(self, X, **kwargs):
        return self.model(X, **kwargs)[1]


def hypothetical_attributions(multipliers, inputs, baselines):
    """A function for aggregating contributions into hypothetical attributions.

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

    projected_contribs = torch.zeros_like(baselines[0], dtype=baselines[0].dtype)

    for i in range(inputs[0].shape[1]):
        hypothetical_input = torch.zeros_like(inputs[0], dtype=baselines[0].dtype)
        hypothetical_input[:, i] = 1.0
        hypothetical_diffs = hypothetical_input - baselines[0]
        hypothetical_contribs = hypothetical_diffs * multipliers[0]

        projected_contribs[:, i] = torch.sum(hypothetical_contribs, dim=1)

    return (projected_contribs,)


params = 'void(int64, int64[:], int64[:], int32[:, :], int32[:,], '
params += 'int32[:, :], float32[:, :, :], int32)'


@numba.jit(params, nopython=False)
def _fast_shuffle(n_shuffles, chars, idxs, next_idxs, next_idxs_counts,
                  counters, shuffled_sequences, random_state):
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

    shuffled_sequences = numpy.zeros((n_shuffles, *sequence.shape),
                                     dtype=numpy.float32)
    counters = numpy.zeros((n_shuffles, len(chars)), dtype=numpy.int32)

    _fast_shuffle(n_shuffles, chars, idxs, next_idxs, next_idxs_counts,
                  counters, shuffled_sequences, random_state)

    shuffled_sequences = torch.from_numpy(shuffled_sequences)
    return shuffled_sequences


def calculate_attributions(model, X, model_output="profile",
                           n_shuffles=20, verbose=False):

    if model_output is None:
        wrapper = model
    elif model_output == "profile":
        wrapper = ProfileWrapper(model)
    elif model_output == "count":
        wrapper = CountWrapper(model)
    else:
        raise ValueError("model_output must be None, 'profile' or 'count'.")

    attributions = []
    references_ = []
    dev = next(model.parameters()).device

    for i in trange(0, len(X), 1, disable=not verbose):
        _X = X[i].float()

        _references = dinucleotide_shuffle(_X, n_shuffles=n_shuffles).to(dev)
        _references = _references.type(_X.dtype)
        _X = _X.unsqueeze(0).to(dev)
        dl = DeepLiftShap(wrapper)
        attr = dl.attribute(_X, _references,
                            custom_attribution_func=hypothetical_attributions,
                            )
        attributions.append(attr.cpu().detach())

    attributions = torch.cat(attributions)
    return attributions

@torch.no_grad()
def projected_shap(attributions, seqs, bs=64, device='cpu'):
    attributions_projected = []
    for i in range(0, len(attributions), bs):
        _attributions = attributions[i:i + bs].to(device)
        _seqs = seqs[i:i + bs].to(device)
        _attributions_projected = (_attributions * _seqs).sum(dim=1)
        attributions_projected.append(_attributions_projected.detach().cpu().numpy())
    attributions_projected = np.concatenate(attributions_projected)
    return attributions_projected
def attribution_to_bigwig(attributions,
                          regions,
                          chrom_size,
                          mode='average', output='attribution.bigwig'):
    """

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
    regions.columns = ['chrom', 'start', 'end'] + list(regions.columns[3:])

    bw = pyBigWig.open(output, 'w')
    header = []

    for chrom in chrom_size:
        header.append((chrom, chrom_size[chrom]))
    bw.addHeader(header, maxZooms=0)

    regions = bioframe.cluster(regions)
    regions['fake_id'] = np.arange(len(regions))
    regions = regions.sort_values(by=['chrom', 'start', 'end'])
    attributions = attributions[np.array(regions['fake_id'])]
    cluster_stats = regions.drop_duplicates('cluster')
    cluster_stats = cluster_stats.sort_values(by=['chrom', 'start', 'end'])
    order_of_cluster = np.array(cluster_stats['cluster'])

    for cluster in tqdm(order_of_cluster):
        mask = regions['cluster'] == cluster
        attributions_chrom = attributions[mask]
        region_temp = regions[mask]
        chroms, starts, ends = region_temp['chrom'], region_temp['start'], region_temp['end']
        start_point = np.min(starts)
        end_point = np.max(ends)
        size = end_point - start_point + 1
        importance_track = np.full((len(region_temp), size), np.nan)

        ct = 0
        for start, end, attribution in zip(starts, ends, attributions_chrom):
            if end-start != attribution.shape[0]:
                print ("shape incompatible", start, end, attribution.shape)
            importance_track[ct, start-start_point:end-start_point] = attribution
            ct += 1

        importance_track = np.nanmean(importance_track, axis=0)
        indx = ~np.isnan(importance_track)
        chrom = np.array(chroms)[0]
        bw.addEntries(str(chrom),
                      np.where(indx)[0] + start_point,
                      values=importance_track[indx].astype('float'), span=1, )
    bw.close()







