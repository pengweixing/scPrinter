"""
Scripts adapted from 10x motif-matching
Tools for analyzing motifs in the genome.
"""
from __future__ import annotations
import tempfile
import MOODS
import MOODS.scan
import MOODS.tools
import MOODS.parsers
import numpy as np
import pandas as pd
from Bio import motifs
from tqdm.auto import tqdm, trange
from . import genome
from .datasets import JASPAR2022_core, CisBP_Human, CisBPJASPA
from .utils import regionparser
from pyfaidx import Fasta
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from functools import partial
import time
import itertools
from pathlib import Path
from typing_extensions import Literal

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

def thresholds_from_p(m, bg, pvalue):
    return MOODS.tools.threshold_from_p(m, bg, pvalue)


class PFM:
    def __init__(self,
                 name,
                 counts):
        self.name = name
        self.counts = counts
        self.length = len(counts['A'])


def parse_jaspar(file_path):
    # Initialize variables
    records = []
    record = None

    # Open the file for reading
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces
            if line.startswith('>'):  # Record name line
                # Save the previous record if it exists
                if record:
                    records.append(PFM(record['name'], record['weights']))
                # Start a new record
                record = {'name': line[1:].replace(' ', ''), 'weights': {'A': [], 'C': [], 'G': [], 'T': []}}
            elif len(line) > 0:
                # Parse the weight line
                nucleotide, values_str = line.split(' ', 1)
                values = list(map(float, values_str.strip('[]').split()))
                if record:
                    record['weights'][nucleotide] = values

        # Don't forget to add the last record after exiting the loop
        if record:
            records.append(PFM(record['name'], record['weights']))

    return records


def _jaspar_to_moods_matrix(jaspar_motif, bg, pseudocount, mode='motifmatchr'):

    with (tempfile.NamedTemporaryFile() as fn):
        f = open(fn.name, "w")
        for base in 'ACGT':
            line = " ".join(str(x) for x in jaspar_motif.counts[base])
            f.write(line + "\n")

        f.close()
        if mode != 'motifmatchr':
            m = MOODS.parsers.pfm_to_log_odds(fn.name, bg, pseudocount, 2)
        else:
            even = [0.25, 0.25, 0.25, 0.25]
            m =  MOODS.parsers.pfm_to_log_odds(fn.name, even, pseudocount, 2)
            m = [tuple(m[i] - (np.log2(0.25) - np.log2(bg[i]))) for i in range(len(bg))]
        return m


def global_scan(seq, motifs_length, motifs_name, chr, start, end, peak_idx, tfs, clean, split_tfs, strand):
    global scanner
    scan_res = scanner.scan(str(seq))
    results = [[[xxx.pos, xxx.score] for xxx in xx] for xx in scan_res]

    return _parse_scan_results_all(results,
                                   motifs_length, motifs_name,
                                   {'chr': chr, 'start': start, 'end': end, 'index': peak_idx},
                                   tfs, clean, split_tfs, strand)


def _parse_scan_results_all(moods_scan_res,
                            motifs_length,
                            motifs_name,
                            bed_coord,
                            tfs,
                            clean=False,
                            split_tf=True,
                            strand_spec=True):
    all_hits = [] # final return, lists of hits in this region.

    # First group by motif.name
    tf2results = {}

    for (motif_idx, hits) in enumerate(moods_scan_res):

        motif_name = motifs_name[motif_idx % len(motifs_name)]
        motif_name = motif_name.split(",") if split_tf else [motif_name]
        motif_length = motifs_length[motif_idx % len(motifs_name)]
        strand = -1 if motif_idx >= len(motifs_name) else 1
        if not strand_spec:
            strand = '*'
        for name in motif_name:
            if name not in tfs:
                continue
            if name not in tf2results:
                tf2results[name] = {}
            if strand not in tf2results[name]:
                tf2results[name][strand] = []
            v = []
            for h in hits:
                motif_start = int(h[0])
                motif_end = int(h[0]) + motif_length
                score = round(h[1], 4)
                # If direct output, then organize the record directly without merge overlapping hits
                if not clean:
                    record = [bed_coord['chr'], bed_coord['start'],
                              bed_coord['end'], bed_coord['index'],
                              name, score,
                              strand, motif_start, motif_end]
                    all_hits.append(record)
                else:

                    motif_start = int(h[0])
                    motif_end = int(h[0]) + motif_length
                    v.append([motif_start, motif_end])
            if clean:
                tf2results[name][strand] += v
    if not clean:
        return all_hits

    for name in tf2results:
        for strand in tf2results[name]:
            v  = tf2results[name][strand]
            if len(v) == 0:
                continue
            v = [np.arange(s,e) for s,e in v]
            v = np.unique(np.concatenate(v).reshape((-1)))
            v = np.sort(v)
            v = consecutive(v)
            for h in v:
                motif_start = np.min(h)
                motif_end = np.max(h)
                record = [bed_coord['chr'], bed_coord['start'],
                          bed_coord['end'], bed_coord['index'],
                          name, 0,
                          strand, motif_start, motif_end]

                all_hits.append(record)
    return all_hits




def _parse_scan_results_clean(moods_scan_res, motifs, bed_coord, tfs):
    """ Parse results of MOODS.scan.scan_dna and return/write
        The default input contains one pair of a single motif: forward and reverse strand
        out_format: if "bed" each hit will be the motif position in bed format, with the start and end as peak coordinates
                    if "motif" same as "bed" except the start and end columns are the motif location
                    if "position" each hit will be a list of positions within a peak region (relative to the start position)
                    if "count" each hit will be an integer of the number of occurrences. NO OUTPUT if count == 0
                    if "binary" each hit will be 1 (any occurrence).  NO OUTPUT if count == 0
    """

    all_hits = []
    # First group by motif.name

    scan_dict = {}
    # print (len(moods_scan_res), len(motifs))


    for (motif_idx, hits) in enumerate(moods_scan_res):

        motif = motifs[motif_idx % len(motifs)]
        # strand = -1 if motif_idx >= len(motifs) else 1
        for name in motif.name.split(","):
            if name not in tfs:
                continue
            if name not in scan_dict:
                scan_dict[name] = []
            v = []
            for h in hits:
                motif_start = int(h.pos)
                motif_end = int(h.pos) + motif.length
                v.append(np.arange(motif_start, motif_end))

            scan_dict[name] += v

    for name in scan_dict:
        v  = scan_dict[name]
        if len(v) == 0:
            continue
        v = np.unique(np.concatenate(v).reshape((-1)))
        v = np.sort(v)
        v = consecutive(v)
        for h in v:
            motif_start = np.min(h)
            motif_end = np.max(h)
            record = [bed_coord['chr'], bed_coord['start'], bed_coord['end'], name, 0,
                      1, motif_start, motif_end]

            all_hits.append(record)


    return all_hits


def _parse_scan_results(moods_scan_res, motifs, bed_coord, tfs):
    """ Parse results of MOODS.scan.scan_dna and return/write
        The default input contains one pair of a single motif: forward and reverse strand
        out_format: if "bed" each hit will be the motif position in bed format, with the start and end as peak coordinates
                    if "motif" same as "bed" except the start and end columns are the motif location
                    if "position" each hit will be a list of positions within a peak region (relative to the start position)
                    if "count" each hit will be an integer of the number of occurrences. NO OUTPUT if count == 0
                    if "binary" each hit will be 1 (any occurrence).  NO OUTPUT if count == 0
    """

    all_hits = []
    for (motif_idx, hits) in enumerate(moods_scan_res):

        motif = motifs[motif_idx % len(motifs)]
        strand = -1 if motif_idx >= len(motifs) else 1

        if len(hits) > 0:
            for h in hits:
                motif_start = int(h.pos)
                motif_end = int(h.pos) + motif.length
                score = round(h.score, 4)
                for name in motif.name.split(","):
                    if name not in tfs:
                        continue
                    record = [bed_coord['chr'], bed_coord['start'], bed_coord['end'], bed_coord['index'],
                              name, score,
                              strand, motif_start, motif_end]

                    all_hits.append(record)
    all_hits = np.unique(np.array(all_hits), axis=0)
    return all_hits

class Motifs:
    """
    A class for motif matching based on MOODS

    Parameters
    ----------
    ref_path_motif : str | Path
        Path to the motif file, in JASPAR format
    ref_path_fa : str | Path
        Path to the reference genome fasta file
    bg : Literal['even', 'genome'] | tuple, optional
        Background nucleotide frequency, by default 'even' stands for equal frequency of A, C, G, T
        'genome' stands for the frequency of A, C, G, T in the reference genome
        When passing a tuple of size 4, they would be used as background frequency
    pseudocount : float, optional
        Pseudocount for the motif, by default 0.8 (same as motifmatchR)
    pvalue : float, optional
        P-value threshold for motif matching, by default 5e-5
    nCores : int, optional
        Number of cores to use, by default 32
    """
    def __init__(self,
                 ref_path_motif: str | Path,
                 ref_path_fa: str | Path,
                 bg: Literal['even', 'genome'] | tuple = 'even',
                 pseudocount: float = 0.8,
                 pvalue: float = 5e-5,
                 nCores: int = 32,
                 split_tf: bool = True,
                 mode: Literal['motifmatchr', 'moods'] = 'motifmatchr'
                 ):
        self.n_jobs = nCores
        self.mode = mode
        # self.pool = ProcessPoolExecutor(max_workers=nCores)
        # with open(ref_path_motif, "r") as infile:
        #     self.all_motifs = list(motifs.parse(infile, "jaspar"))
        self.all_motifs = parse_jaspar(ref_path_motif)
        # self.all_motifs = np.array(self.all_motifs,dtype=object)
        self.names = [set(motif.name.split(",")) if split_tf else {motif.name} for motif in self.all_motifs]
        self.tfs = set().union(*self.names)
        # for large sequence header, only keep the text before the first space
        self.genome_seq = Fasta(ref_path_fa)
        if bg == 'even':
            self.bg = [0.25, 0.25, 0.25, 0.25]
        elif bg == 'genome':
            b = ""
            for chrom in self.genome_seq.keys():
                b += str(self.genome_seq[chrom])

            self.bg = MOODS.tools.bg_from_sequence_dna(str(b), 1)
        else:
            self.bg = bg
        self.pre_matrices_p, \
            self.pre_thresholds_p, \
            self.pre_matrices_m, \
            self.pre_thresholds_m = self._prepare_moods_settings(self.all_motifs, self.bg, pseudocount, pvalue)
        self.pseudocount = pseudocount

        self.pvalue = pvalue


    def prep_scanner(self,
                     tf_genes: list[str] | None = None,
                     pseudocount: float = 0.8,
                     pvalue: float = 5e-5,
                     window: int = 7
                     ):
        """
        Prepare the MOODS scanner for motif matching

        Parameters
        ----------
        tf_genes: list[str] | None
            List of TFs to be used for motif matching, by default None, which means all TFs
        pseudocount: float
            Pseudocount for the motif, by default 0.8 (same as motifmatchR)
        pvalue: float
            P-value threshold for motif matching, by default 5e-5
        window: int
            Window size for motif matching, by default 7 (same as motifmatchR), passed to MOODS

        Returns
        -------
        scanner: MOODS.scan.Scanner

        """
        global scanner
        if tf_genes is None:
            tf_genes = self.tfs
        tf_genes_set = set(tf_genes)
        select = slice(None) if tf_genes is None else [len(name & tf_genes_set) > 0 for name in self.names]
        self.select = select

        if pseudocount != self.pseudocount or pvalue != self.pvalue:
            # motif = self.all_motifs[select]
            motif = [motif for motif, keep in zip(self.all_motifs, select) if keep]
            # Each TF gets a matrix for the + and for the - strand, and a corresponding threshold
            matrices_p, threshold_p, matrices_m, threshold_m = self._prepare_moods_settings(motif, self.bg, pseudocount, pvalue)
        else:

            matrices_p, threshold_p, matrices_m, threshold_m = self.pre_matrices_p[select], \
                                                                self.pre_thresholds_p[select], \
                                                                self.pre_matrices_m[select], \
                                                                self.pre_thresholds_m[select]

        matrices = np.concatenate([matrices_p, matrices_m])
        thresholds = np.concatenate([threshold_p, threshold_m])

        scanner_ = MOODS.scan.Scanner(window)  # parameter is the window size
        scanner_.set_motifs(matrices, self.bg, thresholds)
        self.scanner = scanner_
        scanner = scanner_
        self.tfs = set(tf_genes)
        # print ("finish preparing scanner")
        return scanner_

    def scan_once(self, chr, start, end,
                  clean: bool = False,
                  concat: bool = True,
                  split_tfs: bool = True,
                  strand: bool = True,
                  ):
        global scanner
        scanner = self.scanner
        # motif = self.all_motifs[self.select]
        motif = [motif for motif, keep in zip(self.all_motifs, self.select) if keep]
        motifs_length = [m.length for m in motif]
        motifs_name = [m.name for m in motif]
        seq = self.genome_seq[chr][start:end].seq.upper()

        # seq is of unicode format, need to convert to str
        scan_res = scanner.scan(str(seq))
        results = [[[xxx.pos, xxx.score] for xxx in xx] for xx in scan_res]

        return _parse_scan_results_all(results,
                                   motifs_length, motifs_name,
                                   {'chr': chr, 'start': start, 'end': end, 'index':0},
                                   self.tfs, clean, split_tfs, strand)


    def chromvar_scan(self, adata,
                      verbose: bool = True):
        peaks = regionparser(list(adata.var_names))
        res = self.scan_motif(peaks, verbose=verbose, count=True)
        res = np.array(res)
        res = pd.DataFrame(res, index=adata.var_names,
                           columns=[m.name for m in self.all_motifs])
        adata.varm['motif_match'] = res.to_numpy()
        adata.uns['motif_name'] = res.columns

    def scan_motif(self, peaks_iter,
                    clean: bool = False,
                    concat: bool = True,
                   verbose: bool = False,
                   split_tfs: bool = True,
                    strand: bool = True,
                   count: bool = False,
                    ):
        """
        Scan motifs in the peaks iterator

        Parameters
        ----------
        peaks_iter:
            peaks iterator, yielding a list [chrom, start, end], e.g. a pandas dataframe with three columns.
            This is one of the few functions in scprinter that doesn't support various region formats.
        clean: bool
            Whether to clean the output, by default False.
            If True, overlapping motif hits of the same TF will be merged into one hit, making it better for
            visualization.
            When False, the output will be the raw output from MOODS, which is a list of motif hits for each TF.
        concat: bool
            Whether to concatenate the output, by default True.
            If False, will return the list the same length of the number of peaks,
            each element is a list of motif hits for each TF.

        Returns
        -------

        """
        global scanner
        scanner = self.scanner

        maps = [[] for i in range(len(peaks_iter))]
        # motif = self.all_motifs[self.select]
        motif = [motif for motif, keep in zip(self.all_motifs, self.select) if keep]
        motifs_length = [m.length for m in motif]
        motifs_name = [m.name for m in motif]
        name2id = {name: i for i, name in enumerate(motifs_name)}


        peaks_iter = np.array(peaks_iter)
        if verbose: bar = trange(len(peaks_iter) * 2)
        # results_all = []
        pool = ProcessPoolExecutor(max_workers=self.n_jobs)
        p_list = []
        for peak_idx, peak in enumerate(peaks_iter):
            # peak = peaks_iter[peak_idx]
            chr = peak[0]
            start = int(peak[1])
            end = int(peak[2])
            seq = self.genome_seq[chr][start:end].seq.upper()

            # seq is of unicode format, need to convert to str
            # scan_res = scanner.scan(str(seq))
            # bar.update(1)
            # results = [[[xxx.pos, xxx.score] for xxx in xx] for xx in scan_res]
            #
            # p_list.append(self.pool.submit(_parse_scan_results_all, results,
            #                                 motifs_length, motifs_name,
            #                                 {'chr': chr, 'start': start, 'end': end, 'index': peak_idx},
            #                                 self.tfs, clean, split_tfs, strand))
            p_list.append(pool.submit(global_scan,
                                           seq,
                                           motifs_length,
                                           motifs_name,
                                           chr, start, end,
                                           peak_idx,
                                           self.tfs, clean, split_tfs, strand))

            if len(p_list) >= (self.n_jobs * 10):
                for p in as_completed(p_list):
                    if verbose:
                        bar.update(1)
                    all_hits = p.result()

                    if len(all_hits) > 0:
                        idx = all_hits[0][3]
                        if not count:
                            maps[idx] = all_hits
                        else:
                            freq = np.zeros((1,len(motifs_length)))
                            for hit in all_hits:
                                freq[0, name2id[hit[4]]] = 1
                            maps[idx] = freq
                    p_list.remove(p)
                    del p

                    if len(p_list) <= self.n_jobs:
                        break

            if verbose:
                bar.update(1)
        for p in as_completed(p_list):
            if verbose:
                bar.update(1)
            all_hits = p.result()
            if len(all_hits) > 0:
                idx = all_hits[0][3]
                if not count:
                    maps[idx] = all_hits
                else:
                    freq = np.zeros((1,len(motifs_length)))
                    for hit in all_hits:
                        freq[0, name2id[hit[4]]] = 1
                    maps[idx] = freq
        if count:
            for i in range(len(maps)):
                if len(maps[i]) == 0:
                    maps[i] = np.zeros((1,len(motifs_length)))

        pool.shutdown(wait=True)
        # maps = list(self.pool.map(_parse_scan_results, results_all)) if not clean else list(self.pool.map(_parse_scan_results_clean, results_all))
        if verbose:
            bar.close()
        if concat:
            return list(itertools.chain.from_iterable(maps))
        return maps

    def _prepare_moods_settings(self,
                                jaspar_motifs,
                                bg,
                                pseduocount,
                                pvalue=1e-5):
        """Find hits of list of jaspar_motifs in pyfasta object fasta, using the background distribution bg and
        pseudocount, significant to the give pvalue
        """
        start = time.time()
        pool = ProcessPoolExecutor(max_workers=self.n_jobs)
        matrices_p = list(pool.map(partial(_jaspar_to_moods_matrix, bg=bg, pseudocount=pseduocount, mode=self.mode),
                                        jaspar_motifs, chunksize=100))
        matrices_m = list(pool.map(MOODS.tools.reverse_complement, matrices_p, chunksize=100))
        # print('get_mtx', time.time() - start)
        start = time.time()


        thresholds_p = list(pool.map(partial(thresholds_from_p, bg=bg, pvalue=pvalue), matrices_p, chunksize=100))
        thresholds_m = thresholds_p if self.mode == 'motifmatchr' else list(pool.map(partial(thresholds_from_p, bg=bg, pvalue=pvalue), matrices_m, chunksize=100))
        # thresholds_p = [MOODS.tools.threshold_from_p(m, bg, pvalue) for m in matrices_p]
        # thresholds_m = [MOODS.tools.threshold_from_p(m, bg, pvalue) for m in matrices_m]
        # print('get_thresholds', time.time() - start)
        start = time.time()
        # print (thresholds_p, thresholds_m)
        pool.shutdown(wait=True)
        # self.pool = ProcessPoolExecutor(max_workers=self.n_jobs)
        return np.array(matrices_p,dtype=object), np.array(thresholds_p,dtype=object), \
            np.array(matrices_m,dtype=object), np.array(thresholds_m,dtype=object)

def JASPAR2022_core_Motifs(genome: genome.Genome, bg, **kwargs):
    """
    Jaspar2022 core motifs

    Parameters
    ----------
    genome
    bg

    Returns
    -------

    """
    if bg == 'genome':
        bg = genome.bg
    return Motifs(JASPAR2022_core(), genome.fetch_fa(), bg, **kwargs)


def CisBPHuman_Motifs(genome: genome.Genome, bg, **kwargs):
    """
    CisBP Human motifs

    Parameters
    ----------
    genome
    bg

    Returns
    -------

    """
    if bg == 'genome':
        bg = genome.bg
    return Motifs(CisBP_Human(), genome.fetch_fa(), bg, **kwargs)

def CisBP_JASPAR_Motifs(genome: genome.Genome, bg, **kwargs):
    """
    CisBP + JASPAR motifs (will have redundant motif hits!)
    Only use when the TF you study have really bad motifs

    Parameters
    ----------
    genome
    bg

    Returns
    -------

    """
    if bg == 'genome':
        bg = genome.bg
    return Motifs(CisBPJASPA(), genome.fetch_fa(), bg, **kwargs)

