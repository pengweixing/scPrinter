import os
import sys

import numpy as np
import torch
import pandas
import pyfaidx
import pyBigWig
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset
from ..utils import DNA_one_hot

class ChromBPDataset(torch.utils.data.Dataset):
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

    def __init__(self, signals,
                 ref_seq,
                 summits,
                 DNA_window,
                 signal_window,
                 max_jitter,
                 min_counts,
                 max_counts,
                 cached=False,
                 lazy_cache=False,
                 reverse_compliment=False,
                 device='cpu',
                 initialize=True):
        self.signal_paths = signals
        self.signals = [pyBigWig.open(signal) for signal in signals]
        self.ref_seq_path = ref_seq
        self.ref_seq = pyfaidx.Fasta(ref_seq)
        self.chrom_size = {k: len(v[:].seq) for k, v in self.ref_seq.items()}
        # print ("chrom_size")
        # for k,v in self.chrom_size.items():
        #     if "_" not in k:
        #         print (k,":",v)

        self.max_jitter = max_jitter
        self.min_counts = min_counts
        self.max_counts = max_counts
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
        if self.min_counts is None:
            self.min_counts = -1
        if self.max_counts is None:
            self.max_counts = 1e16

        # summits.iloc[:, 1] = summits.iloc[:, 1].astype(int)
        if initialize:
            print("input summits", len(summits))
            summits_valid = np.array([self.validate_loci(chrom,
                                                         summit) for
                                      chrom, summit in
                                      zip(tqdm(summits.iloc[:, 0], desc='validating loci'),
                                          summits.iloc[:, 1])])
            print("valid summits after trimming edges", np.sum(summits_valid))
            self.summits = summits.loc[summits_valid]

            coverage = np.array([self.fetch_cov(chrom, summit) for
                                 chrom, summit in
                                 zip(tqdm(self.summits.iloc[:, 0], desc='fetching coverage'),
                                     self.summits.iloc[:, 1])])
            print (coverage.shape)
            print ("coverage min max", coverage.min(axis=-1).min(), coverage.max(axis=-1).max())
            summits_valid = (coverage.min(axis=-1) > self.min_counts) & (coverage.max(axis=-1) < self.max_counts)
            print("valid summits after min/max count filter", np.sum(summits_valid))
            self.summits = self.summits.loc[summits_valid]
            self.coverage = coverage[summits_valid]
            self.summits = np.array(self.summits)
            if lazy_cache:
                self.cache_seqs = None
                self.cache_signals = None
                self.uncache_index = set(np.arange(len(self.summits)))

            if self.cached:
                self.cache_seqs = []
                self.cache_signals = []
                for chrom, summit in tqdm(self.summits[:, :2], desc='Caching sequences'):
                    DNA, signal = self.fetch_loci(chrom, summit)
                    self.cache_seqs.append(DNA)
                    self.cache_signals.append(signal)
                self.cache_seqs = torch.stack(self.cache_seqs, dim=0)
                self.cache_signals = torch.stack(self.cache_signals, dim=0)



    def change_jitter(self, max_jitter):
        self.max_jitter = max_jitter
        summits_valid = np.array([self.validate_loci(chrom,
                                                     summit) for
                                  chrom, summit in
                                  zip(tqdm(self.summits[:, 0], desc='validating loci'),
                                      self.summits[:, 1])])
        print("valid summits after trimming edges", np.sum(summits_valid))
        self.summits = self.summits[summits_valid]

        self.coverage = np.array([self.fetch_cov(chrom, summit) for
                                  chrom, summit in
                                  zip(tqdm(self.summits[:, 0], desc='fetching coverage'),
                                      self.summits[:, 1])])

        if self.cached:
            self.cache(force=True)

    def cache(self, force=False):
        if self.cached and not force:
            print("Already cached")
            return

        self.cached = True
        self.cache_seqs = []
        self.cache_signals = []
        for chrom, summit in tqdm(self.summits[:, :2], desc='Caching sequences'):
            DNA, signal = self.fetch_loci(chrom, summit)
            self.cache_seqs.append(DNA)
            self.cache_signals.append(signal)
        self.cache_seqs = torch.stack(self.cache_seqs, dim=0)
        self.cache_signals = torch.stack(self.cache_signals, dim=0)

    def append(self, dataset):
        self.summits = np.concatenate([self.summits, dataset.summits])
        self.coverage = np.concatenate([self.coverage, dataset.coverage])
        if self.cached or dataset.cached:
            self.cache()
            dataset.cache()
            self.cache_seqs = torch.cat([self.cache_seqs, dataset.cache_seqs])
            self.cache_signals = torch.cat([self.cache_signals, dataset.cache_signals])

    def to(self, device):
        self.device = device
        return self

    def __len__(self):
        return len(self.summits)

    def apply_jitter(self, dna, signal):
        jitter = 0 if self.max_jitter == 0 else np.random.default_rng().integers(self.max_jitter * 2)
        return dna[:, jitter:jitter + self.DNA_window], signal[:, jitter:jitter + self.signal_window]

    def __getitem__(self, idx):
        reshape = False
        if self.cached:
            dnas, signals = (self.cache_seqs[idx].to(self.device),
                             self.cache_signals[idx].to(self.device))
            if len(dnas.shape) == 2:
                reshape = True
                dnas = dnas[None]
                signals = signals[None]

        else:
            # print (self.summits)
            summits = self.summits[idx][..., :2]
            if len(summits.shape) == 1:
                summits = summits[None]
                reshape = True
            dnas, signals = [], []
            for summit in summits:
                chrom, summit = summit
                dna, signal = self.fetch_loci(chrom, summit, device=self.device)

                dnas.append(dna)
                signals.append(signal)
            dnas = torch.stack(dnas, dim=0)
            signals = torch.stack(signals, dim=0)
            if self.lazy_cache:
                if self.cache_seqs is None:
                    shapes = list(dnas.shape)
                    if len(shapes) == 3:
                        shapes[0] = len(self.summits)
                    else:
                        shapes = [len(self.summits)] + shapes
                    self.cache_seqs = torch.zeros(shapes, dtype=torch.float32, device=self.device)
                    shapes = list(signals.shape)
                    if len(shapes) == 3:
                        shapes[0] = len(self.summits)
                    else:
                        shapes = [len(self.summits)] + shapes
                    self.cache_signals = torch.zeros(shapes, dtype=torch.float32, device=self.device)

                self.cache_seqs[idx] = dnas
                self.cache_signals[idx] = signals
                try:
                    self.uncache_index.remove(idx)
                except:
                    for i in idx:
                        self.uncache_index.remove(i)

                if len(self.uncache_index)  == 0:
                    print ("finish caching")
                    self.cached = True

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


        return (dnas.float(), signals)

    def downsample(self, num):
        idx = torch.as_tensor(np.random.permutation(len(self.summits))[:num]).to(self.device)
        summits = self.summits[idx]
        coverage = self.coverage[idx]
        if self.cached:
            cache_seqs = self.cache_seqs[idx]
            cache_signals = self.cache_signals[idx]
        downsampled_dataset = ChromBPDataset(signals=self.signal_paths,
                                             ref_seq=self.ref_seq_path,
                                             summits=summits,
                                             DNA_window=self.DNA_window,
                                             signal_window=self.signal_window,
                                             max_jitter=self.max_jitter,
                                             min_counts=self.min_counts,
                                             max_counts=self.max_counts,
                                             cached=self.cached,
                                             reverse_compliment=self.reverse_compliment,
                                             initialize=False,
                                             device=self.device)
        downsampled_dataset.coverage = coverage
        downsampled_dataset.summits = summits
        if self.cached:
            downsampled_dataset.cache_seqs = cache_seqs
            downsampled_dataset.cache_signals = cache_signals
        return downsampled_dataset

    def filter_by_coverage(self, min_coverage, max_coverage):
        valid = (self.coverage.sum(axis=-1) > min_coverage) & (self.coverage.sum(axis=-1) < max_coverage)
        self.summits = self.summits[valid]
        self.coverage = self.coverage[valid]
        self.uncache_index = set(np.arange(len(self.summits)))
        if self.cached:
            self.cache_seqs = self.cache_seqs[valid]
            self.cache_signals = self.cache_signals[valid]

    def fetch_cov(self, chrom, summit):
        cov = [np.nansum(signal.values(chrom,
                                       summit - self.signal_flank,
                                       summit + self.signal_flank, numpy=True))
               for signal in self.signals]
        return np.array(cov)

    def validate_loci(self, chrom, summit):
        if summit - self.max_flank - self.max_jitter < 0:
            return False
        if summit + self.max_flank + self.max_jitter >= self.chrom_size[chrom]:
            return False

        return True

    def fetch_loci(self, chrom, summit,
                   device='cpu'):
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
        DNA = DNA_one_hot(self.ref_seq[chrom][summit - self.DNA_flank - self.max_jitter:
                                              summit + self.DNA_flank + self.max_jitter].seq.upper(),
                          device=device)
        signal = np.zeros((len(self.signals), (self.signal_flank + self.max_jitter) * 2))
        for i, signal_file in enumerate(self.signals):
            try:
                signal[i, :] = np.nan_to_num(signal_file.values(
                    chrom,
                    summit - self.signal_flank - self.max_jitter,
                    summit + self.signal_flank + self.max_jitter))
            except:
                print("signal error",
                      chrom,
                      summit,
                      summit - self.signal_flank - self.max_jitter,
                      summit + self.signal_flank + self.max_jitter
                      )
                raise EOFError
        signal = torch.tensor(signal, dtype=torch.float32, device=device)
        return DNA, signal

def collate_fn_singlecell(batch):
    """A collate function for the single cell data.
    Parameters
    ----------
    batch: list
        A list of tuples of two torch.Tensor objects: DNA sequence and signal track.

    Returns
    -------
    tuple
        A tuple of two torch.Tensor objects: DNA sequence and signal track.
    """

    X, y, cell, peak, pos_mask, coverage = list(zip(*batch))
    X = torch.concat(X, dim=0)
    y = torch.concat(y, dim=0)
    cell = torch.concat(cell, dim=0)[:, None]
    peak = torch.concat(peak, dim=0)[:, None]
    pos_mask = torch.concat(pos_mask, dim=0)
    idx = np.arange(len(y))
    pos = idx[pos_mask]
    neg = idx[~pos_mask]
    ratio = 0.1
    sampled_neg = np.random.permutation(neg)[:int(len(pos) * ratio)]
    idx = torch.as_tensor(np.random.permutation(np.concatenate([pos, sampled_neg])))
    y = y[idx]
    X = X[idx]
    cell = cell[idx]
    peak = peak[idx]
    return X, y, cell, peak, coverage


class ChromBPDataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 dataset=None,
                 batch_size=64,
                 num_workers=1,
                 pin_memory=True,
                 shuffle=True,
                 collate_fn=None,
                 **kwargs):
        self.data_collection = None
        self.batch_size = batch_size
        if dataset is None:
            self.dataset = ChromBPDataset(**kwargs)
        elif type(dataset) is list:
            self.data_collection = dataset
            dataset_list = []
            for data, num in dataset:
                dataset_list.append(data.downsample(num))
            self.dataset = ConcatDataset(dataset_list)

        else:
            self.dataset = dataset
        self.shuffle = shuffle
        super().__init__(self.dataset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         shuffle=shuffle,
                         worker_init_fn = lambda id: np.random.seed(id),
                         collate_fn=collate_fn)

    def resample(self):
        if self.data_collection is not None:
            dataset_list = []
            for data, num in self.data_collection:
                dataset_list.append(data.downsample(num))
            return ChromBPDataLoader(ConcatDataset(dataset_list),
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=self.pin_memory,
                                     shuffle=self.shuffle,)
        else:
            return self



hg38_splits = [None] * 5
hg38_splits[0] = {
        "test": [
            "chr1",
            "chr3",
            "chr6"
        ],
        "valid": [
            "chr8",
            "chr20"
        ],
        "train": [
            "chr2",
            "chr4",
            "chr5",
            "chr7",
            "chr9",
            "chr10",
            "chr11",
            "chr12",
            "chr13",
            "chr14",
            "chr15",
            "chr16",
            "chr17",
            "chr18",
            "chr19",
            "chr21",
            "chr22",
            "chrX",
            "chrY"
        ]
      }
hg38_splits[1] = {
    "test": [
        "chr2",
        "chr8",
        "chr9",
        "chr16"
    ],
    "valid": [
        "chr12",
        "chr17"
    ],
    "train": [
        "chr1",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr10",
        "chr11",
        "chr13",
        "chr14",
        "chr15",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chr22",
        "chrX",
        "chrY"
    ]
  }
hg38_splits[2] = {
    "test": [
        "chr4",
        "chr11",
        "chr12",
        "chr15",
        "chrY"
    ],
    "valid": [
        "chr22",
        "chr7"
    ],
    "train": [
        "chr1",
        "chr2",
        "chr3",
        "chr5",
        "chr6",
        "chr8",
        "chr9",
        "chr10",
        "chr13",
        "chr14",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chrX"
    ]
  }
hg38_splits[3] = {
    "test": [
        "chr5",
        "chr10",
        "chr14",
        "chr18",
        "chr20",
        "chr22"
    ],
    "valid": [
        "chr6",
        "chr21"
    ],
    "train": [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr7",
        "chr8",
        "chr9",
        "chr11",
        "chr12",
        "chr13",
        "chr15",
        "chr16",
        "chr17",
        "chr19",
        "chrX",
        "chrY"
    ]
  }
hg38_splits[4] = {
    "test": [
        "chr7",
        "chr13",
        "chr17",
        "chr19",
        "chr21",
        "chrX"
    ],
    "valid": [
        "chr10",
        "chr18"
    ],
    "train": [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr8",
        "chr9",
        "chr11",
        "chr12",
        "chr14",
        "chr15",
        "chr16",
        "chr20",
        "chr22",
        "chrY"
    ]
  }


mm10_splits = [None] * 5
mm10_splits[0] = {
    "test": [
        "chr1",
        "chr6",
        "chr12",
        "chr13",
        "chr16"
    ],
    "valid": [
        "chr8",
        "chr11",
        "chr18",
        "chr19",
        "chrX"
    ],
    "train": [
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr7",
        "chr9",
        "chr10",
        "chr14",
        "chr15",
        "chr17"
    ]
}
mm10_splits[1] = {
    "test": [
        "chr2",
        "chr7",
        "chr10",
        "chr14",
        "chr17"
    ],
    "valid": [
        "chr5",
        "chr9",
        "chr13",
        "chr15",
        "chrY"
    ],
    "train": [
        "chr1",
        "chr3",
        "chr4",
        "chr6",
        "chr8",
        "chr11",
        "chr12",
        "chr16",
        "chr18",
        "chr19",
        "chrX"
    ]
}
mm10_splits[2] = {
    "test": [
        "chr3",
        "chr8",
        "chr13",
        "chr15",
        "chr17"
    ],
    "valid": [
        "chr2",
        "chr9",
        "chr11",
        "chr12",
        "chrY"
    ],
    "train": [
        "chr1",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr10",
        "chr14",
        "chr16",
        "chr18",
        "chr19",
        "chrX"
    ]
}
mm10_splits[3] = {
    "test": [
        "chr4",
        "chr9",
        "chr11",
        "chr14",
        "chr19"
    ],
    "valid": [
        "chr1",
        "chr7",
        "chr12",
        "chr13",
        "chrY"
    ],
    "train": [
        "chr2",
        "chr3",
        "chr5",
        "chr6",
        "chr8",
        "chr10",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chrX"
    ]
}
mm10_splits[4] = {
    "test": [
        "chr5",
        "chr10",
        "chr12",
        "chr16",
        "chrY"
    ],
    "valid": [
        "chr3",
        "chr7",
        "chr14",
        "chr15",
        "chr18"
    ],
    "train": [
        "chr1",
        "chr2",
        "chr4",
        "chr6",
        "chr8",
        "chr9",
        "chr11",
        "chr13",
        "chr17",
        "chr19",
        "chrX"
    ]
}
