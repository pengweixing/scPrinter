# motivated by snapatac2.genome
from __future__ import annotations

import os.path
from pathlib import Path

import gffutils
import h5py
import numpy as np
import pyBigWig
from pyfaidx import Fasta

from .datasets import datasets, giverightstothegroup
from .utils import DNA_one_hot, get_stats_for_genome


class Genome:
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
       The path to the bias h5 file
    bg: tuple
       A tuple of background frequencies of ACGT
       When setting as None, it will be inferred from the fasta file

    Returns
    -------
    None
    """

    def __init__(
        self,
        name: str,
        chrom_sizes: dict | None = None,
        gff_file: str | Path = "",
        fa_file: str | Path = "",
        bias_file: str | Path = "",
        blacklist_file: str | Path | None = None,
        bg: tuple | None = None,
        splits: list[dict] | None = None,
    ):

        self.name = name
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
        self.blacklist_file = blacklist_file
        self.gff_file = self.fetch_gff()
        self.fa_file = self.fetch_fa()
        self.bias_file = self.fetch_bias()
        self.blacklist_file = self.fetch_blacklist()
        self.bias_bw = self.fetch_bias_bw()
        self.gff_db = self.fetch_gff_db()
        self.splits = splits

    def fetch_blacklist(self):
        """
        Fetch the blacklist file from the server if it is not present locally
        Returns
        -------

        """
        if self.blacklist_file is None:
            raise ValueError("No blacklist file provided")
        if not os.path.exists(self.blacklist_file):
            return str(
                datasets().fetch(
                    self.blacklist_file,
                    processor=giverightstothegroup,
                    progressbar=True,
                )
            )
        else:
            return self.blacklist_file

    def fetch_gff(self):
        """
        Fetch the gff file from the server if it is not present locally

        Returns
        -------
        str of the path to the local gff file
        """
        if not os.path.exists(self.gff_file):
            return str(
                datasets().fetch(self.gff_file, progressbar=True, processor=giverightstothegroup)
            )
        else:
            return self.gff_file

    def fetch_gff_db(self):
        """
        Fetch the gff file locally, if it is not present, create it

        Returns
        -------

        """
        gff = self.fetch_gff()
        if not os.path.exists(str(gff) + ".db"):
            print("Creating GFF database (Runs for new genome)")
            # Specifying the id_spec was necessary for gff files from NCBI.
            gffutils.create_db(
                gff,
                str(gff) + ".db",
                id_spec={"gene": "gene_name", "transcript": "transcript_id"},
                merge_strategy="create_unique",
            )
        return str(gff) + ".db"

    def fetch_fa(self):
        """
        Fetch the fasta file from the server if it is not present locally

        Returns
        -------
        str of the path to the local fasta file
        """
        if not os.path.exists(self.fa_file):
            return str(
                datasets().fetch(
                    self.fa_file,
                    processor=giverightstothegroup,
                    progressbar=True,
                )
            )
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
        if not hasattr(self, "fasta"):
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
        if not hasattr(self, "fasta"):
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
            file = datasets().fetch(
                self.bias_file,
                processor=giverightstothegroup,
                progressbar=True,
            )

            for f in file:
                if ".h5" in f:
                    return str(f)
        return self.bias_file

    def fetch_bias_bw(self):
        """
        Fetch the bias bigwig file locally, if it is not present, create it
        Returns
        -------

        """
        bias_file = self.fetch_bias()
        bias_bw = bias_file.replace(".h5", ".bw")
        if not os.path.exists(bias_bw):
            print("creating bias bigwig (runs for new bias h5 file)")
            with h5py.File(bias_file, "r") as dct:
                precomputed_bias = {chrom: np.array(dct[chrom]) for chrom in dct.keys()}
                bw = pyBigWig.open(bias_bw, "w")
                header = []
                for chrom in precomputed_bias:
                    sig = precomputed_bias[chrom]
                    length = sig.shape[-1]
                    header.append((chrom, length))
                bw.addHeader(header, maxZooms=0)
                for chrom in precomputed_bias:
                    sig = precomputed_bias[chrom]
                    bw.addEntries(
                        str(chrom),
                        np.arange(len(sig)),
                        values=sig.astype("float"),
                        span=1,
                    )
                bw.close()
        return bias_bw


hg38_splits = [None] * 5
hg38_splits[0] = {
    "test": ["chr1", "chr3", "chr6"],
    "valid": ["chr8", "chr20"],
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
        "chrY",
    ],
}
hg38_splits[1] = {
    "test": ["chr2", "chr8", "chr9", "chr16"],
    "valid": ["chr12", "chr17"],
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
        "chrY",
    ],
}
hg38_splits[2] = {
    "test": ["chr4", "chr11", "chr12", "chr15", "chrY"],
    "valid": ["chr22", "chr7"],
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
        "chrX",
    ],
}
hg38_splits[3] = {
    "test": ["chr5", "chr10", "chr14", "chr18", "chr20", "chr22"],
    "valid": ["chr6", "chr21"],
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
        "chrY",
    ],
}
hg38_splits[4] = {
    "test": ["chr7", "chr13", "chr17", "chr19", "chr21", "chrX"],
    "valid": ["chr10", "chr18"],
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
        "chrY",
    ],
}

mm10_splits = [None] * 5
mm10_splits[0] = {
    "test": ["chr1", "chr6", "chr12", "chr13", "chr16"],
    "valid": ["chr8", "chr11", "chr18", "chr19", "chrX"],
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
        "chr17",
    ],
}
mm10_splits[1] = {
    "test": ["chr2", "chr7", "chr10", "chr14", "chr17"],
    "valid": ["chr5", "chr9", "chr13", "chr15", "chrY"],
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
        "chrX",
    ],
}
mm10_splits[2] = {
    "test": ["chr3", "chr8", "chr13", "chr15", "chr17"],
    "valid": ["chr2", "chr9", "chr11", "chr12", "chrY"],
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
        "chrX",
    ],
}
mm10_splits[3] = {
    "test": ["chr4", "chr9", "chr11", "chr14", "chr19"],
    "valid": ["chr1", "chr7", "chr12", "chr13", "chrY"],
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
        "chrX",
    ],
}
mm10_splits[4] = {
    "test": ["chr5", "chr10", "chr12", "chr16", "chrY"],
    "valid": ["chr3", "chr7", "chr14", "chr15", "chr18"],
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
        "chrX",
    ],
}

GRCh38 = Genome(
    "hg38",
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
        "chrY": 57227415,
    },
    "gencode_v41_GRCh38.gff3.gz",
    "gencode_v41_GRCh38.fa.gz",
    "hg38Tn5Bias.tar.gz",
    "hg38-blacklist.v2.bed.gz",
    (0.29518279760588795, 0.20390602956403897, 0.20478356895235347, 0.2961276038777196),
    hg38_splits,
)

hg38 = GRCh38


GRCm38 = Genome(
    "mm10",
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
    "mm10-blacklist.v2.bed.gz",
    (0.29149763779592625, 0.2083275235867118, 0.20834346947899296, 0.291831369138369),
    mm10_splits,
)

mm10 = GRCm38
