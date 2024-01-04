# motivated by snapatac2.genome
from __future__ import annotations
import os.path
from .datasets import datasets, giverightstothegroup
from .utils import get_stats_for_genome, DNA_one_hot
from pooch import Decompress, Untar
from pathlib import Path
from pyfaidx import Fasta
import torch




class Genome:
    def __init__(self,
                 chrom_sizes: dict | None = None,
                 gff_file: str | Path = '',
                 fa_file: str | Path = '',
                 bias_file: str | Path = '',
                 bg: tuple | None = None):
        """
        Genome class
        It keep records of all the genome specific information
        including chrom sizes, gff gene annotation file,
        genome reference fasta file,
        precomputed Tn5 bias file, and background freq of ACGT

        Parameters
        ----------
        chrom_sizes: dict
            A dictionary of chromosome names and their lengths
            When setting as None, it will be inferred from the fasta file
        gff_file: str
            The path to the gff file
        fa_file: str
            The path to the fasta file
        bias_file: str
            The path to the bias file
        bg: tuple
            A tuple of background frequencies of ACGT
            When setting as None, it will be inferred from the fasta file

        Returns
        -------
        None
        """

        if chrom_sizes is None or bg is None:
            chrom_sizes_inferred, bg_inferred = get_stats_for_genome(fa_file)
            if chrom_sizes is None:
                self.chrom_sizes = chrom_sizes_inferred
            else:
                self.chrom_sizes = chrom_sizes
            if bg is None:
                self.bg = bg_inferred
            else:
                self.bg = bg
        else:
            self.chrom_sizes = chrom_sizes
            self.bg = bg
        self.gff_file = gff_file
        self.fa_file = fa_file
        self.bias_file = bias_file

    def fetch_gff(self):
        """
        Fetch the gff file from the server if it is not present locally

        Returns
        -------
        str of the path to the local gff file
        """
        if not os.path.exists(self.gff_file):
            return str(datasets().fetch(self.gff_file, progressbar=True, processor=giverightstothegroup))
        else:
            return self.gff_file

    def fetch_fa(self):
        """
        Fetch the fasta file from the server if it is not present locally

        Returns
        -------
        str of the path to the local fasta file
        """
        if not os.path.exists(self.fa_file):
            return str(datasets().fetch(self.fa_file, processor = giverightstothegroup, progressbar=True,))
        else:
            return self.fa_file

    def fetch_seq(self, chrom, start, end):
        """
        Fetch the sequence from the fasta file

        Parameters
        ----------
        chrom: str
            The name of the chromosome
        start: int
            The start position of the sequence
        end: int
            The end position of the sequence

        Returns
        -------
        str of the sequence
        """
        if not hasattr(self, 'fasta'):
            self.fasta = Fasta(self.fetch_fa())

        return self.fasta[chrom][start:end].seq

    def fetch_onehot_seq(self, chrom, start, end):
        """
        Fetch the onehot encoded sequence from the fasta file

        Parameters
        ----------
        chrom: str
            The name of the chromosome
        start: int
            The start position of the sequence
        end: int
            The end position of the sequence

        Returns
        -------
        np.array of the onehot encoded sequence
        """
        if not hasattr(self, 'fasta'):
            self.fasta = Fasta(self.fetch_fa())

        return DNA_one_hot(self.fasta[chrom][start:end].seq.upper())

    def fetch_bias(self):
        """
        Fetch the bias file from the server if it is not present locally

        Returns
        -------
        str of the path to the local bias file
        """
        if not os.path.exists(self.bias_file):
            return str(datasets().fetch(self.bias_file, processor = giverightstothegroup, progressbar=True,)[0])
        return self.bias_file


GRCh38 = Genome(
    {
        "chr1": 248956422,
        "chr2": 242193529,
        "chr3": 198295559,
        "chr4": 190214555,
        "chr5": 181538259,
        "chr6": 170805979,
        "chr7": 159345973,
        "chr8": 145138636,
        "chr9": 138394717,
        "chr10": 133797422,
        "chr11": 135086622,
        "chr12": 133275309,
        "chr13": 114364328,
        "chr14": 107043718,
        "chr15": 101991189,
        "chr16": 90338345,
        "chr17": 83257441,
        "chr18": 80373285,
        "chr19": 58617616,
        "chr20": 64444167,
        "chr21": 46709983,
        "chr22": 50818468,
        "chrX": 156040895,
        "chrY": 57227415
    },
    "gencode_v41_GRCh38.gff3.gz",
    "gencode_v41_GRCh38.fa.gz",
    "hg38Tn5Bias.tar.gz",
    (0.29518279760588795, 0.20390602956403897, 0.20478356895235347, 0.2961276038777196)
)

hg38 = GRCh38


GRCm38 = Genome(
    {
        "chr1": 195471971,
        "chr2": 182113224,
        "chr3": 160039680,
        "chr4": 156508116,
        "chr5": 151834684,
        "chr6": 149736546,
        "chr7": 145441459,
        "chr8": 129401213,
        "chr9": 124595110,
        "chr10": 130694993,
        "chr11": 122082543,
        "chr12": 120129022,
        "chr13": 120421639,
        "chr14": 124902244,
        "chr15": 104043685,
        "chr16": 98207768,
        "chr17": 94987271,
        "chr18": 90702639,
        "chr19": 61431566,
        "chrX": 171031299,
        "chrY": 91744698,
    },
    "gencode_vM25_GRCm38.gff3.gz",
    "gencode_vM25_GRCm38.fa.gz",
    "mm10Tn5Bias.tar.gz",
    (0.29149763779592625,
     0.2083275235867118,
     0.20834346947899296,
     0.291831369138369)
)

mm10 = GRCm38

