# adapted from bpnetlite - attribution.py

import time
import numpy
import numba
import torch
import torch.nn as nn
import pandas
import logomaker
import pyBigWig
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from tqdm.auto import trange
from .utils import DNA_one_hot
import numpy as np
from tqdm.auto import tqdm, trange
import bioframe
from captum.attr import IntegratedGradients, InputXGradient, Saliency
from .captum_deeplift import DeepLiftShap
import sys
from shap import DeepExplainer
from .shap_deeplift import PyTorchDeep
from typing import Literal
from .ism import ism
class SignalWrapperFootprint(torch.nn.Module):
    def __init__(self, model, nth_output=0,
                 activation=None,
                 transformation=None,
                 res=1):
        super().__init__()
        self.model = model
        self.nth_output = nth_output
        self.activation = activation
        self.transformation = transformation
        self.res = res
        self.sniff_padding('N' * 2114)

    @torch.no_grad()
    def sniff_padding(self, X):
        dev = next(self.model.parameters()).device
        if type(X) is str:
            X = DNA_one_hot(X.upper())
        X = X.float().to(dev)[None]
        logits = self.model(X)[:, self.nth_output]
        # print (X.shape, logits.shape)
        self.padding = (X.shape[-1] - logits.shape[-1] * self.res) // 2

    def forward(self, X, *args, **kwargs):
        dev = next(self.model.parameters()).device
        if type(X) is str:
            X = DNA_one_hot(X.upper())
        X = X.float().to(dev)[None]
        logits = self.model(X, *args, **kwargs)[:, self.nth_output]
        if self.activation is not None:
            logits = self.activation(logits)
        if self.transformation is not None:
            w, b = self.transformation
            logits = logits * w + b
        logits = logits[0]
        return logits

class _ProfileLogitScaling(torch.nn.Module):
	"""This ugly class is necessary because of Captum.

	Captum internally registers classes as linear or non-linear. Because the
	profile wrapper performs some non-linear operations, those operations must
	be registered as such. However, the inputs to the wrapper are not the
	logits that are being modified in a non-linear manner but rather the
	original sequence that is subsequently run through the model. Hence, this
	object will contain all of the operations performed on the logits and
	can be registered.


	Parameters
	----------
	logits: torch.Tensor, shape=(-1, -1)
		The logits as they come out of a Chrom/BPNet model.
	"""

	def __init__(self):
		super(_ProfileLogitScaling, self).__init__()

	def forward(self, logits):
		y = torch.nn.functional.log_softmax(logits, dim=-1)
		y = logits * torch.exp(y).detach()
		return y

class InverseSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x = torch.clip(x, 1e-6, 1-1e-6)
        # return x * torch.log(x / (1 - x))
        return x * torch.sigmoid(x)

class ProfileWrapperFootprintClass(torch.nn.Module):
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

    def __init__(self, model, nth_output=0, res=1):
        super().__init__()
        self.model = model
        self.nth_output = nth_output
        self.res = res
        # self.scaling = _ProfileLogitScaling()
        # self.sniff_padding('N' * 2114)
        # self.relu = nn.ReLU()
        self.scaling = InverseSigmoid()
        # self.scaling = nn.Sigmoid()
    @torch.no_grad()
    def sniff_padding(self, X):
        dev = next(self.model.parameters()).device
        if type(X) is str:
            X = DNA_one_hot(X.upper())
        X = X.float().to(dev)[None]
        logits = self.model(X)[:, self.nth_output]
        # print(X.shape, logits.shape)
        self.padding = (X.shape[-1] - logits.shape[-1] * self.res) // 2

    def forward(self, X, *args, **kwargs):
        logits = self.model(X, *args, **kwargs)[:, self.nth_output]
        logits = logits.reshape(X.shape[0], -1)
        # y = logits
        logits = logits - torch.mean(logits, dim=-1, keepdims=True)
        #
        # y = self.scaling(logits)
        # y = self.relu(logits)
        y = self.scaling(logits)
        # y = logits
        return (y).mean(axis=-1, keepdims=True)
        # return self.relu(logits).sum(axis=-1, keepdims=True)
        # return logits.mean(axis=-1, keepdims=True)

class ProfileWrapperFootprint(torch.nn.Module):
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

    def __init__(self, model, nth_output=0, res=1):
        super().__init__()
        self.model = model
        self.nth_output = nth_output
        self.res = res
        self.scaling = _ProfileLogitScaling()
        # self.sniff_padding('N' * 2114)
        # self.relu = nn.ReLU()
        # self.scaling = InverseSigmoid()

    @torch.no_grad()
    def sniff_padding(self, X):
        dev = next(self.model.parameters()).device
        if type(X) is str:
            X = DNA_one_hot(X.upper())
        X = X.float().to(dev)[None]
        logits = self.model(X)[:, self.nth_output]
        # print(X.shape, logits.shape)
        self.padding = (X.shape[-1] - logits.shape[-1] * self.res) // 2

    def forward(self, X, *args, **kwargs):
        logits = self.model(X, *args, **kwargs)[:, self.nth_output]
        logits = logits.reshape(X.shape[0], -1) #/ 10 + 0.5
        logits = logits - torch.mean(logits, dim=-1, keepdims=True)
        #
        # y = self.scaling(logits)
        # y = self.relu(logits)
        y = self.scaling(logits)
        # y = logits
        return (y).sum(axis=-1, keepdims=True)
        # return self.relu(logits).sum(axis=-1, keepdims=True)
        # return logits.mean(axis=-1, keepdims=True)



    def attribute(self, seq, additional_forward_args=None):
        dev = next(self.model.parameters()).device
        if type(seq) is str:
            seq = DNA_one_hot(seq.upper())
        _X = seq.float()
        _references = dinucleotide_shuffle(_X, n_shuffles=20).to(dev)
        _references = _references.type(_X.dtype)
        _X = _X.unsqueeze(0).to(dev)
        dl = DeepLiftShap(self)
        attr = dl.attribute(_X, _references,
                            custom_attribution_func=hypothetical_attributions,
                            additional_forward_args=(additional_forward_args, )
                            )
        attr = (attr * _X)[0].sum(dim=0).detach()
        return attr


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
        logits = (logits - torch.mean(logits, dim=-1, keepdims=True))

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
    # print (multipliers.shape, inputs.shape, baselines.shape)
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

    shuffled_sequences = torch.from_numpy(shuffled_sequences).float()
    return shuffled_sequences


def calculate_attributions(model, X, model_output="profile",
                           method = 'deeplift',
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

    if method == 'deeplift' or method == 'deeplift_hypo':
        dl = DeepLiftShap(wrapper)
    elif method == 'integrated_gradients' or method == 'integrated_gradients_hypo':
        dl = IntegratedGradients(wrapper)
    elif method == 'ism':
        for i in trange(0, len(X), 32, disable=not verbose):
            _X = X[i:i+32].float().to(dev)
            wrapper.zero_grad()
            attr = ism(wrapper, _X, batch_size=128, verbose=True)
            attributions.append(attr.cpu().detach())

        attributions = torch.cat(attributions)
        return attributions

    for i in trange(0, len(X), 1, disable=not verbose):
        _X = X[i].float()
        wrapper.zero_grad()
        _references = dinucleotide_shuffle(_X, n_shuffles=n_shuffles).to(dev)
        _references = _references.type(_X.dtype)
        _X = _X.unsqueeze(0).to(dev)
        if method == 'deeplift' or method == 'deeplift_hypo':
            attr,delta = dl.attribute(_X, _references,
                                custom_attribution_func=hypothetical_attributions if "hypo" in method else None,
                                additional_forward_args=(None),
                                return_convergence_delta=True,
                                )
            if 'hypo' not in method:
                print (delta)
            # raise EOFError
        elif method == 'integrated_gradients' or method == 'integrated_gradients_hypo':
            # dl = IntegratedGradients(wrapper)
            _X = torch.cat([_X] * n_shuffles, dim=0)
            # print (_X.shape, _references.shape)
            # attrs = []
            # for ref in _references:
            #     attr = dl.attribute(_X, baselines=ref[None])
            #     # attr = hypothetical_attributions(attr, _X, ref)
            #     attrs.append(attr.detach())
            # attrs = torch.cat(attrs, dim=0)
            # "deeplift","shap", "deeplift_hypo", "shap_hypo", "inputxgradient",
            attrs, delta = dl.attribute(_X,
                                        # baselines=torch.ones_like(_X) * 0.25,
                                        baselines=_references,
                                        internal_batch_size=50,
                                        n_steps=10,
                                        return_convergence_delta=True)
            if torch.max(torch.abs(delta)[0]) > 1e-2:
                print (delta)
            if 'hypo' in method:
                attrs, = hypothetical_attributions((attrs,),
                                                  (_X,),
                                                  (_references,))
                attr = attrs.mean(dim=0, keepdims=True)
            else:
                attr = attrs.mean(dim=0, keepdims=True)
            # print(attrs.shape, _references.shape, torch.cat([_X] * n_shuffles, dim=0))
            #
        elif method == 'shap' or method == 'shap_hypo':
            # _references = torch.zeros_like(_X)
            dl = PyTorchDeep(wrapper, _references)
            # dl = DeepExplainer(wrapper, _references)
            # _X = torch.cat([_X] * n_shuffles, dim=0)
            attr, deltas = dl.shap_values(_X, check_additivity=False,
                                        custom_attribution_func=hypothetical_attributions if "hypo" in method else None)#.to(dev)[0].mean(dim=0)
            # attr = torch.from_numpy(attr)
            if "hypo" not in method:
                print (deltas)

        elif 'inputxgradient' in method:
            # dl = InputXGradient(wrapper.forward)
            # attr = dl.attribute(_X)
            # dl = Saliency(wrapper)
            wrapper.zero_grad()
            _X = _X * 0.99 + (0.01) / 4 * torch.ones_like(_X)
            _X.requires_grad = True
            with torch.autograd.set_grad_enabled(True):
                output = wrapper(_X)
                attr = torch.autograd.grad(output, _X)[0]
                print (attr.shape, torch.min(attr), torch.max(attr), torch.norm(attr))
            # attr = dl.attribute(_X, abs=False)
            # print (attr.shape)
            if 'norm' in method:
                attr = attr - (attr.sum(dim=1, keepdims=True) - attr)/3
            if 'abs' in method:
                attr = torch.abs(attr)



        else:
            raise ValueError(f"method {method} not supported")
        # if i >= 10:
        #     raise EOFError

        if len(attr.shape) == 2:
            attr = attr[None]

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
#
def attribution_to_bigwig(attributions,
                          regions,
                          chrom_size,
                           res=1,
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


    regions = bioframe.cluster(regions)
    regions['fake_id'] = np.arange(len(regions))
    regions = regions.sort_values(by=['chrom', 'start', 'end'])
    # attributions = attributions[np.array(regions['fake_id'])]
    cluster_stats = regions.drop_duplicates('cluster')
    cluster_stats = cluster_stats.sort_values(by=['chrom', 'start', 'end'])
    order_of_cluster = np.array(cluster_stats['cluster'])

    clusters = {cluster: group for cluster, group in regions.groupby('cluster')}

    header = []
    chrom_order = cluster_stats['chrom'].unique()
    for chrom in chrom_order:
        header.append((chrom, int(chrom_size[chrom])))
    bw.addHeader(header, maxZooms=10)

    for cluster in tqdm(order_of_cluster):
        region_temp = clusters[cluster]
        chroms, starts, ends = region_temp['chrom'], region_temp['start'], region_temp['end']
        mask = np.array(region_temp['fake_id'])
        attributions_chrom = attributions[mask]


        if len(mask) == 1:
            chrom = np.array(chroms)[0]
            # print (str(chrom),
            #               np.arange(np.array(starts)[0],np.array(ends)[0]),
            #               attributions_chrom[0].astype('float'))
            points = np.arange(attributions_chrom[0].shape[-1]) * res + np.array(starts)[0]
            # print('only 1', attributions_chrom.shape, points.shape)
            bw.addEntries(str(chrom),
                          points,
                          values=attributions_chrom[0].astype('float'), span=res,)
            continue

        start_point = np.min(starts)
        end_point = np.max(ends)
        size = (end_point - start_point) // res
        importance_track = np.full((len(region_temp), size), np.nan)
        # print (mask, attributions_chrom.shape, importance_track.shape)
        ct = 0
        for start, end, attribution in zip(starts, ends, attributions_chrom):
            if end-start != attribution.shape[0] * res:
                print ("shape incompatible", start, end, attribution.shape)
            importance_track[ct, (start-start_point) // res:(end-start_point) // res] = attribution
            ct += 1

        importance_track = np.nanmean(importance_track, axis=0)
        indx = ~np.isnan(importance_track)
        chrom = np.array(chroms)[0]
        points = np.array(np.where(indx)[0]) * res + start_point
        bw.addEntries(str(chrom),
                      points,
                      values=importance_track[indx], span=res, )
    bw.close()
