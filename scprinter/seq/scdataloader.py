import multiprocessing as mpl
import os
import sys
from functools import partial

import numpy as np
import pandas
import pyBigWig
import pyfaidx
import torch
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map

from ..utils import DNA_one_hot


def fetch_cov(summit, group2idx):
    global global_insertion_dict
    chrom, start, end = summit
    signal = global_insertion_dict[chrom][:, start:end].sum(axis=-1)
    coverages = np.zeros((len(group2idx),))
    for j in range(len(group2idx)):
        coverages[j] = signal[group2idx[j]].sum()
    return coverages


class scChromBPDataset(torch.utils.data.Dataset):
    """A general Data generator for training and evaluation.
       It takes in a bigwig, a list of summits, and some parameters specifying the length of the sequences
       the length of outputs etc. and generate training data on the fly.

    Parameters
    ----------
    signal: str
        Path to the bigwig file.
    ref_seq: str
        Path to the reference genome fasta file.
    summits: pandas.DataFrame
        A dataframe containing the summits of the peaks.
    DNA_window: int
        Length of the input DNA sequence. The sequence will be centered at the summit.
    signal_window: int
        Length of the output signal track. The signal sequence will be centered at the summit.
    """

    def __init__(
        self,
        insertion_dict,
        bias,
        group2idx,
        ref_seq,
        summits,
        DNA_window,
        signal_window,
        max_jitter,
        cached=False,
        lazy_cache=False,
        reverse_compliment=False,
        device="cpu",
        initialize=True,
        coverages=None,
        data_augmentation=False,
        mode="uniform",
        cell_sample=10,
    ):
        global global_insertion_dict
        global_insertion_dict = insertion_dict
        # self.insertion_dict = insertion_dict
        self.bias_path = bias
        self.bias = pyBigWig.open(bias)
        self.group2idx = group2idx
        self.ref_seq_path = ref_seq
        self.ref_seq = pyfaidx.Fasta(ref_seq)
        self.chrom_size = {k: len(v[:].seq) for k, v in self.ref_seq.items()}

        self.max_jitter = max_jitter
        self.reverse_compliment = reverse_compliment
        self.DNA_window = DNA_window
        self.signal_window = signal_window
        self.DNA_flank = DNA_window // 2
        self.signal_flank = signal_window // 2
        self.max_flank = max(self.DNA_flank, self.signal_flank)
        self.cached = cached
        self.device = device
        self.summits = summits
        self.lazy_cache = lazy_cache
        self.data_augmentation = data_augmentation
        self.mode = mode
        self.cell_sample = cell_sample

        if initialize:
            print("input summits", len(summits))
            summits_valid = np.array(
                [
                    self.validate_loci(chrom, summit)
                    for chrom, summit in zip(
                        tqdm(summits.iloc[:, 0], desc="validating loci"),
                        summits.iloc[:, 1],
                    )
                ]
            )
            print("valid summits after trimming edges", np.sum(summits_valid))
            self.summits = summits.loc[summits_valid]
            self.summits = self.summits.loc[summits_valid]
            self.summits = np.array(self.summits)

            self.idx = np.arange(len(self.summits) * len(self.group2idx))
            self.summit_idx = self.idx // len(self.group2idx)
            self.group_idx = self.idx % len(self.group2idx)

            func = partial(fetch_cov, group2idx=self.group2idx)
            regions = [
                self.summits[:, 0],
                self.summits[:, 1] - self.signal_flank - self.max_jitter,
                self.summits[:, 1] + self.signal_flank + self.max_jitter,
            ]
            regions = np.array(regions).T
            print(regions, regions.shape)
            regions[:, 1] = regions[:, 1].astype(int)
            regions[:, 2] = regions[:, 2].astype(int)
            if coverages is None:
                coverages = process_map(
                    func,
                    regions,
                    max_workers=min(mpl.cpu_count(), 128),
                    chunksize=100,
                    desc="fetching coverages",
                )
                self.coverages = np.array(coverages).reshape((-1))
            else:
                self.coverages = coverages
                assert len(self.coverages) == len(self.summits) * len(
                    self.group2idx
                ), "coverages length mismatch"

            print(
                "coverages",
                self.coverages.shape,
                self.coverages.min(),
                self.coverages.max(),
            )
            self.normalized_coverages = self.coverages.reshape((-1, len(self.group2idx)))
            self.normalized_coverages = np.log1p(
                self.normalized_coverages / self.normalized_coverages.mean(axis=1)[:, None]
            )
            print(
                "normalized coverages",
                self.normalized_coverages.shape,
                self.normalized_coverages.min(),
                self.normalized_coverages.max(),
            )
            self.normalized_coverages = torch.from_numpy(
                self.normalized_coverages.reshape((-1))
            ).float()
            self.mask = self.coverages > 10
            pos = self.idx[self.mask]
            neg = self.idx[~self.mask]
            if data_augmentation:
                ratio = 0.1
                sampled_neg = np.random.permutation(neg)[: int(len(pos) * ratio)]
                self.idx = np.concatenate([pos, sampled_neg])
                self.idx = self.idx
            else:
                self.idx = pos

            self.summit_idx = self.summit_idx[self.idx]
            self.group_idx = self.group_idx[self.idx]
            print("filtering from ", len(self.mask), "to ", len(self.idx))
            if lazy_cache:
                self.cache_seqs = None
                self.cache_bias = None
                self.uncache_index = set(np.arange(len(self.summits)))

            if self.cached:
                self.cache_seqs = []
                self.cache_bias = []
                for chrom, summit in tqdm(self.summits[:, :2], desc="Caching sequences"):
                    DNA, bias = self.fetch_loci_dna_bias(chrom, summit)
                    self.cache_seqs.append(DNA)
                    self.cache_bias.append(bias)

                self.cache_seqs = torch.stack(self.cache_seqs, dim=0)
                self.cache_bias = torch.stack(self.cache_bias, dim=0)

    def cache(self, force=False):
        if self.cached and not force:
            print("Already cached")
            return

        self.cached = True
        self.cache_seqs = []
        self.cache_bias = []
        for chrom, summit in tqdm(self.summits, desc="Caching sequences"):
            DNA, bias = self.fetch_loci_dna_bias(chrom, summit)
            self.cache_seqs.append(DNA)
            self.cache_bias.append(bias)

        self.cache_seqs = torch.stack(self.cache_seqs, dim=0)
        self.cache_bias = torch.stack(self.cache_bias, dim=0)

    def to(self, device):
        self.device = device
        return self

    def __len__(self):
        if self.mode == "uniform":
            return len(self.idx)
        elif self.mode == "peak":
            return len(self.summits)
        else:
            raise ValueError("mode not recognized")

    def apply_jitter(self, dna, signal):
        if self.max_jitter == 0:
            return dna, signal
        jitter = (
            0 if self.max_jitter == 0 else np.random.default_rng().integers(self.max_jitter * 2)
        )

        return (
            dna[:, jitter : jitter + self.DNA_window],
            signal[:, jitter : jitter + self.signal_window],
        )

    def getitem_by_summit_group(self, summit_idx, group_idx):
        # DNA first
        reshape = False
        if self.cached:
            dnas = self.cache_seqs[summit_idx].to(self.device)
            biases = self.cache_bias[summit_idx].to(self.device)
            if len(dnas.shape) == 2:
                reshape = True
                dnas = dnas[None]
                biases = biases[None]
        else:
            summits = self.summits[summit_idx][..., :2]
            if len(summits.shape) == 1:
                summits = summits[None]
                reshape = True
            dnas = []
            biases = []
            for summit in summits:
                chrom, summit = summit
                dna, bias = self.fetch_loci_dna_bias(chrom, summit, device=self.device)
                dnas.append(dna)
                biases.append(bias)

            dnas = torch.stack(dnas, dim=0)
            biases = torch.stack(biases, dim=0)
            if self.lazy_cache:
                if self.cache_seqs is None:
                    shapes = list(dnas.shape)
                    if len(shapes) == 3:
                        shapes[0] = len(self.summits)
                    else:
                        shapes = [len(self.summits)] + shapes
                    self.cache_seqs = torch.zeros(shapes, dtype=torch.float32, device=self.device)
                    shapes = list(biases.shape)
                    if len(shapes) == 3:
                        shapes[0] = len(self.summits)
                    else:
                        shapes = [len(self.summits)] + shapes
                    self.cache_bias = torch.zeros(shapes, dtype=torch.float32, device=self.device)

                self.cache_seqs[summit_idx] = dnas.float()
                self.cache_bias[summit_idx] = biases
                try:
                    if summit_idx in self.uncache_index:
                        self.uncache_index.remove(summit_idx)
                except:
                    for i in summit_idx:
                        if i in self.uncache_index:
                            self.uncache_index.remove(i)

                if len(self.uncache_index) == 0:
                    print("finish caching")
                    self.cached = True

        # Now for signals
        signals = []
        summits = self.summits[summit_idx][..., :2]
        if len(summits.shape) == 1:
            summits = summits[None]
            group_idx = [group_idx]
            summit_idx = [summit_idx]
            reshape = True
        for summit, group in zip(summits, group_idx):
            chrom, summit = summit
            signal = self.fetch_loci_signal(chrom, summit, device=self.device)
            signal = self.group_signal(signal, group, device=self.device)
            signals.append(signal)
        signals = torch.stack(signals, dim=0)
        signals = torch.concatenate([signals, biases], dim=1)

        dnas_final, signals_final = [], []
        for dna, signal in zip(dnas, signals):
            dna, signal = self.apply_jitter(dna, signal)
            if self.reverse_compliment:
                if np.random.default_rng().random() > 0.5:
                    dna = dna.flip(dims=(0, 1))
                    signal = signal.flip(dims=(1,))

            dnas_final.append(dna)
            signals_final.append(signal)

        dnas, signals = torch.stack(dnas_final, dim=0), torch.stack(signals_final, dim=0)
        if reshape:
            dnas = dnas[0]
            signals = signals[0]

        dnas = dnas.float()
        pos_mask = torch.as_tensor(
            self.mask[summit_idx * len(self.group2idx) + group_idx], dtype=torch.bool
        )
        return (
            dnas.float(),
            signals,
            torch.as_tensor(group_idx).long(),
            torch.as_tensor(summit_idx).long(),
            pos_mask,
        )

    def __getitem__(self, idx):

        if self.mode == "uniform":
            summit_idx = self.summit_idx[idx]
            group_idx = self.group_idx[idx]
            X, y, cell, peak, _ = self.getitem_by_summit_group(summit_idx, group_idx)
            coverage = self.normalized_coverages[self.idx[idx]]
            return X, y, cell, peak, coverage
        elif self.mode == "peak":
            summit_idx = idx
            rng = np.random.default_rng()
            group_idx = rng.integers(low=0, high=len(self.group2idx), size=self.cell_sample)
            grid1, grid2 = np.meshgrid(summit_idx, group_idx)
            combinations = np.column_stack((grid1.flatten(), grid2.flatten()))
            summit_idx, group_idx = combinations[:, 0], combinations[:, 1]

            X, y, cell, peak, pos_mask = self.getitem_by_summit_group(summit_idx, group_idx)
            coverage = self.normalized_coverages[summit_idx * len(self.group2idx) + group_idx]
            return X, y, cell, peak, pos_mask, coverage

    def validate_loci(self, chrom, summit):
        if summit - self.max_flank - self.max_jitter < 0:
            return False
        if summit + self.max_flank + self.max_jitter >= self.chrom_size[chrom]:
            return False

        return True

    def group_signal(self, signal, group=slice(None), device="cpu"):
        signal = np.array(signal[self.group2idx[group]].sum(axis=0)).astype("float32")
        signal = torch.tensor(signal, dtype=torch.float32, device=device)
        return signal

    def fetch_loci_dna_bias(self, chrom, summit, device="cpu"):
        """Fetch the DNA sequence and the signal track for a given locus.
        Parameters
        ----------
        chrom: str
            Chromosome name.
        start: int
            Start position of the locus.
        end: int
            End position of the locus.

        Returns
        -------
        tuple
            A tuple of two torch.Tensor objects: DNA sequence and signal track.
        """
        DNA = DNA_one_hot(
            self.ref_seq[chrom][
                summit
                - self.DNA_flank
                - self.max_jitter : summit
                + self.DNA_flank
                + self.max_jitter
            ].seq.upper(),
            device=device,
        )
        signal = np.nan_to_num(
            self.bias.values(
                chrom,
                summit - self.signal_flank - self.max_jitter,
                summit + self.signal_flank + self.max_jitter,
                numpy=True,
            ),
            nan=0.0,
        )[None]
        signal = torch.tensor(signal, dtype=torch.float32, device=device)

        return DNA, signal

    def fetch_loci_signal(self, chrom, summit, device="cpu"):
        global global_insertion_dict
        # still sparse, and is at single cell resolution
        signal = global_insertion_dict[chrom][
            :,
            summit
            - self.signal_flank
            - self.max_jitter : summit
            + self.signal_flank
            + self.max_jitter,
        ]
        return signal
