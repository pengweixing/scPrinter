
import os
import sys

import numpy as np
import torch
import pandas

import pyfaidx
import pyBigWig
from tqdm.auto import tqdm

from .utils import *


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
                 cached=False,
                 device='cpu'):
        self.signals = [pyBigWig.open(signal) for signal in signals]
        self.ref_seq = pyfaidx.Fasta(ref_seq)
        self.chrom_size = {k:len(v[:].seq) for k,v in self.ref_seq.items()}
        # print ("chrom_size")
        # for k,v in self.chrom_size.items():
        #     if "_" not in k:
        #         print (k,":",v)

        self.max_jitter = max_jitter
        self.DNA_window = DNA_window
        self.signal_window = signal_window
        self.DNA_flank = DNA_window // 2
        self.signal_flank = signal_window // 2
        self.max_flank = max(self.DNA_flank, self.signal_flank)
        self.cached = cached
        self.device = device

        summits_valid = np.array([self.validate_loci(chrom, summit) for
                                  chrom, summit in
                                  zip(summits.iloc[:, 0], summits.iloc[:, 1])])
        self.summits = np.array(summits.loc[summits_valid])
        print("input summits", len(summits))
        print("valid summits", len(self.summits))

        if self.cached:
            self.cache_seqs = []
            self.cache_signals = []
            for chrom, summit in tqdm(self.summits, desc='Caching sequences'):
                DNA, signal = self.fetch_loci(chrom, summit)
                DNA, signal = self.apply_jitter(DNA, signal)
                self.cache_seqs.append(DNA)
                self.cache_signals.append(signal)
            self.cache_seqs = torch.stack(self.cache_seqs, dim=0)
            self.cache_signals = torch.stack(self.cache_signals, dim=0)

    def to(self, device):
        self.device = device
        return self

    def __len__(self):
        return len(self.summits)

    def apply_jitter(self, dna, signal):
        jitter = 0 if self.max_jitter == 0 else np.random.default_rng().integers(self.max_jitter * 2)
        return dna[:, jitter:jitter+self.DNA_window], signal[:, jitter:jitter+self.signal_window]



    def __getitem__(self, idx):
        if self.cached:
            dnas, signals =  (self.cache_seqs[idx].to(self.device),
                    self.cache_signals[idx].to(self.device))
        else:
            summits = self.summits[idx]
            reshape = False
            if len(summits.shape) == 1:
                summits = summits[None]
                reshape = True
            dnas, signals = [], []
            for summit in summits:
                chrom, summit = summit
                dna, signal =  self.fetch_loci(chrom, summit, device=self.device)
                dna, signal = self.apply_jitter(dna, signal)
                dnas.append(dna)
                signals.append(signal)
            dnas = torch.stack(dnas, dim=0)
            signals = torch.stack(signals, dim=0)
            if reshape:
                dnas = dnas[0]
                signals = signals[0]

        return (dnas.float(), signals)

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
        DNA = DNA_one_hot(self.ref_seq[chrom][summit-self.DNA_flank-self.max_jitter:
                                              summit+self.DNA_flank+self.max_jitter].seq.upper(),
                          device=device)
        signal = np.zeros((len(self.signals), (self.signal_flank + self.max_jitter) * 2))
        for i, signal_file in enumerate(self.signals):
            try:
                signal[i, :] = np.nan_to_num(signal_file.values(
                    chrom,
                    summit-self.signal_flank-self.max_jitter,
                    summit+self.signal_flank+self.max_jitter))
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

    
class ChromBPDataLoader(torch.utils.data.DataLoader):

    def __init__(self,
                 batch_size,
                 num_workers,
                 pin_memory,
                 shuffle,
                 signals,
                 ref_seq,
                 summits,
                 DNA_window,
                 signal_window,
                 max_jitter,
                 cached=False,
                 device='cpu'
                 ):
        self.dataset = ChromBPDataset(signals,
                                      ref_seq,
                                      summits,
                                      DNA_window,
                                      signal_window,
                                      max_jitter,
                                      cached,
                                      device)
        super().__init__(self.dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=shuffle)
