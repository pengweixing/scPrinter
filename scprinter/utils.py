from __future__ import annotations
import h5py
import pyranges
import torch
import snapatac2 as snap
import numpy as np
import os
import sys
import gzip
import pandas as pd
from copy import deepcopy
import re
import MOODS
from pyfaidx import Fasta
import pybedtools
from pathlib import Path
from scipy.sparse import csr_matrix


def get_stats_for_genome(fasta_file):
    """
    This is a function that reads a fasta file and return a dictionary of chromosome names and their lengths
    It also counts the frequency of ACGT in the genome.
    You can use this to calculate stats for your own genome.Genome object

    Parameters
    ----------
    fasta_file: str
        Path to the fasta file

    """
    genome_seq = Fasta(fasta_file)
    b = ""
    chrom_sizes = {}
    for chrom in genome_seq.keys():
        s = str(genome_seq[chrom])
        b += s
        chrom_sizes[chrom] = len(s)
    bg = MOODS.tools.bg_from_sequence_dna(str(b), 1)
    return chrom_sizes, bg


def get_genome_bg(genome):
    genome_seq = Fasta(genome.fetch_fa())
    b = ""
    for chrom in genome_seq.keys():
        b += str(genome_seq[chrom])
    bg = MOODS.tools.bg_from_sequence_dna(str(b), 1)
    return bg

# a function to parse all kinds of regions specification
def regionparser(regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str] ,
                 printer=None, width:int|None=None):
    """
    This function parses the regions specification and returns a dataframe with the first three columns ['Chromosome', 'Start', 'End']
    This is the base function that makes scPrinter compatible with different regions specification.

    Parameters
    ----------
    regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str]
        - a pandas dataframe with the first three columns correspond to [chrom, start, end],
        - a pyranges object of the regions,
        - or a list of region identifiers, e.g. `['chr1:1000-2000', 'chr1:3000-4000']`
        - or a list of gene identifiers, e.g. `['Gene:CTCF', 'Gene:GATA1']`,
        the transcript start site will be used as the region center

    printer: PyPrinter
        The printer object that contains the gff information. It is required if the regions are specified by gene names.
    width: int | None
        The width of the regions. When specified, the regions will be resized to the specified width.
        If None, the width will be the same as the input regions, and would be 1000bp when regions are specified by gene names.
    """
    if type(regions) is list:
        if ":" in regions[0] and "-" in regions[0]:
            regions = pd.DataFrame([re.split(':|-', xx) for xx in regions], columns=['Chromosome', 'Start', 'End'])
            regions['Start'] = regions['Start'].astype('int')
            regions['End'] = regions['End'].astype('int')
        elif "Gene:" in regions[0]:
            regions = [printer.gff_db[xx[5:]] for xx in regions]
            chrom, start = [xx.chrom for xx in regions], [xx.start if xx.strand == '+' else xx.end for xx in regions]
            regions = pd.DataFrame({'Chromosome':chrom, 'Start': start})
            regions['End'] = regions['Start'] + int(printer.gene_region_width / 2)
            regions['Start'] -= int(printer.gene_region_width / 2)
        # regions_pr = dftopyranges(regions)
    elif type(regions) is pd.core.frame.DataFrame:
        regions = regions
        # regions_pr = dftopyranges(regions)
    elif type(regions) is pd.core.series.Series:
        regions = pd.DataFrame(regions.values[None])
    elif type(regions) is str:
        if ":" in regions and "-" in regions:
            regions = pd.DataFrame([re.split(':|-', regions)], columns=['Chromosome', 'Start', 'End'])
            regions['Start'] = regions['Start'].astype('int')
            regions['End'] = regions['End'].astype('int')
            # regions_pr = dftopyranges(regions)
        elif "Gene:" in regions:
            regions = printer.gff_db[regions[5:]]
            regions = pd.DataFrame({'Chromosome': [regions.chrom], 'Start': [regions.start]})
            regions['End'] = regions['Start'] + int(printer.gene_region_width / 2)
            regions['Start'] -= int(printer.gene_region_width / 2)
        else:
            # regions_pr = pyranges.readers.read_bed(regions)
            regions = pd.read_csv(regions, sep='\t', header=None)

    else:
        print ("Expecting type of list, pd.dataframe, str. Got: ",type(regions), regions)

    regions.columns = ['Chromosome', 'Start', 'End'] + list(regions.columns)[3:]
    if width is not None:
        regions = resize_bed_df(regions, width, True)
    return regions

def frags_to_insertions(data):
    x = data.obsm['fragment_paired']
    insertion = csr_matrix((np.ones(len(x.indices) * 2, dtype='uint16'),
                            np.concatenate([x.indices, x.indices + x.data], axis=0),
                            x.indptr * 2), shape=x.shape)
    insertion.sort_indices()
    insertion.sum_duplicates()
    data.obsm['insertion'] = insertion
    return data

def check_snap_insertion(shift_left=0, shift_right=0):
    temp_fragments = gzip.open("temp_fragments.tsv.gz", "wt")
    # This checks the end to see if snapatac2 now would do insertion at end, or end-1
    for i in range(100):
        temp_fragments.write("chr1\t%d\t%d\tbarcode1\t1\n" % (4, 100))
    temp_fragments.close()
    data = snap.pp.import_data("temp_fragments.tsv.gz",
                           chrom_sizes=snap.genome.hg38.chrom_sizes,
                           min_num_fragments=0,
                           shift_left=shift_left,
                           shift_right=shift_right,
                           # file='testabcdefg.h5ad'
                               )
    data = frags_to_insertions(data)
    v = np.array(data.obsm['insertion'][0, :200].toarray()).reshape((-1))
    os.remove("temp_fragments.tsv.gz")
    # If true: The fixed version I compiled or they fixed it, else not fixed
    return v[100] == 100


def check_snap_insertion_old(shift_left=0, shift_right=0):
    temp_fragments = gzip.open("temp_fragments.tsv.gz", "wt")
    for i in range(100):
        temp_fragments.write("chr1\t%d\t%d\tbarcode1\n" % (4, 4))
    temp_fragments.close()
    sys.stdout.flush()
    data = snap.pp.import_data("temp_fragments.tsv.gz",
                               genome=snap.genome.hg38, min_num_fragments=0, min_tsse=0,
                               shift_left=shift_left,
                               shift_right=shift_right)
    v = np.array(data.obsm['insertion'][0, :10].toarray()).reshape((-1))
    os.remove("temp_fragments.tsv.gz")
    # If true: The fixed version I compiled or they fixed it, else not fixed
    return v[4] == 200


def create_consistent_frags(pathToFrags, extra_plus_shift, extra_minus_shift):
    flag_ = check_snap_insertion()
    output = "temp.fragments.tsv.gz"
    if flag_ and extra_plus_shift == 0 and extra_minus_shift == 0:
        if len(pathToFrags) == 1 :
            return pathToFrags[0], False
        else:
            command = "cat %s > %s" %(" ".join(pathToFrags), output)
            print (command)
            os.system(command)
            # for path in pathToFrags:
            #     command = 'sort -k4, 4 %s > %s' % (path, output)
            #     os.system(command)
            return output, True
    else:
        plus1_flag = 0 if flag_ else 1
        print ("Fixing offset & merging fragments")
        for path in pathToFrags:
            if '.gz' in path:
                csv_file = gzip.open(path, 'rb')
            else:
                csv_file = open(path, 'r')

            reader = pd.read_csv(csv_file, chunksize=100000000, sep="\t", header=None)
            # chrom, start, end, fragments
            # read and process chunk by chunk
            for chunk_fragments in reader:
                chunk_fragments[1] += extra_plus_shift
                chunk_fragments[2] += extra_minus_shift + plus1_flag
                chunk_fragments.to_csv(output, index=False, header=None, sep='\t', mode='a')
        return output, True


def split_insertion_profile(csr_matrix_insertion, chrom, start, end, to_csc=False):
    # print ("Processing insertion profile")
    if to_csc:
        return {chrom: csr_matrix_insertion[:, start:end].tocsc() for chrom, start, end in zip(
            chrom,
            start,
            end)}
    else:
        return  {chrom: csr_matrix_insertion[:, start:end] for chrom,start, end in zip(
        chrom,
        start,
        end)}


def cell_grouping2cell_grouping_idx(printer,
                                    cell_grouping: list[list[str]] | np.ndarray):
    """
    Convert a cell grouping from string based to index based.
    E.g., you pass a list of lists of barcodes `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`,
     and you get a list of lists of indices [[0], [1,2,3]] which corresponds to the index in the single cell insertion profile.

    Parameters
    ----------
    printer: PyPrinter
        The Pyprinter object

    cell_grouping: list[list[str]] | np.ndarray
        The cell grouping in string based format
    """
    # concatenate all the embeddings
    barcodes_all = np.concatenate(cell_grouping)
    # get the unique ones
    uniq = np.unique(barcodes_all)
    # turn barcodes into index in the insertion sparse matrix
    uniq_ix = np.array(printer.insertion_file.obs_ix(np.array(uniq)))
    # construct a mapping from uniq barcode name to its index
    dict1 = {b: i for i, b in zip(uniq_ix, uniq)}
    # and use that to turn string based matching to index based matching
    cell_grouping_idx = [
        [dict1[b] for b in barcodes]
        for barcodes in cell_grouping
    ]
    return cell_grouping_idx

def df2cell_grouping(printer, barcodeGroups):
    """
    Convert a dataframe of barcodes and their groupings (The input format for R's implementation)
    to a list of lists of barcodes (which is scPrinter's input format)

    Parameters
    ----------
    printer: PyPrinter
        The Pyprinter object
    barcodeGroups: pd.DataFrame
        The dataframe of barcodes and their groupings. It needs to at least have two columns,
        where the first column is the barcode and the second column is the group name.

    Returns
    -------
    grouping: list[list[str]]
        The list of lists of barcodes
    uniq_groups: np.ndarray
        The unique group names

    """
    set1_ = set(printer.insertion_file.obs_names)
    mask = [xx in set1_ for xx in barcodeGroups.iloc[:, 0]]
    barcodeGroups = barcodeGroups.iloc[mask,]

    grouping = []
    bar = np.array(barcodeGroups.iloc[:, 0])
    groups = np.array(barcodeGroups.iloc[:, 1])
    uniq_groups = np.unique(groups)
    for group in uniq_groups:
        grouping.append(list(bar[groups == group]))
    return grouping, uniq_groups

def valide_regions(bed: pd.DataFrame, genome, inplace=True):

    chromsize = genome.chrom_sizes
    valid = []
    for chrom, start, end in zip(bed.iloc[:, 0], bed.iloc[:, 1], bed.iloc[:, 2]):
        if chrom not in chromsize:
            valid.append(False)
        elif start < 0 or end > chromsize[chrom]:
            valid.append(False)
        else:
            valid.append(True)
    valid = np.array(valid)
    if inplace:
        return bed.loc[valid, :]
    else:
        return valid


def resize_bed_df(bed: pd.DataFrame,
                  width:int=1000,
                  copy:bool=True,
                  center: str | None=None):
    """
    Resize a bed dataframe to a given width,

    Parameters
    ----------
    bed: pd.DataFrame
        The bed dataframe with the first three columns being chromosome, start, end
    width: int
        The width of the resized bed dataframe
    copy: bool
        Whether to make a copy of the input bed dataframe or change them inplace

    Returns
    -------
    bed: pd.DataFrame
        The resized bed dataframe

    """
    bed = deepcopy(bed) if copy else bed
    if center is None:
        center = np.floor((bed.iloc[:, 1] + bed.iloc[:, 2]) / 2).astype('int')
    
    bed.iloc[:, 1] = np.floor(center - width / 2).astype('int')
    bed.iloc[:, 2] = np.floor(center + width / 2).astype('int')
    return bed

def merge_overlapping_bed(bed: pd.DataFrame,
                          min_requierd_bp:int = 250,
                          copy:bool = True):
    """
    Merge overlapping entries in a bed dataframe

    Parameters
    ----------
    bed: pd.DataFrame
        The bed dataframe with the first three columns being chromosome, start, end
    min_requierd_bp: int
        The minimum number of bp required to merge two entries
    copy: bool
        Whether to make a copy of the input bed dataframe or change them inplace

    Returns
    -------
    bed: pd.DataFrame
        The merged bed dataframe
    """
    bed = deepcopy(bed) if copy else bed
    bed = pybedtools.BedTool.from_dataframe(bed)
    bed = bed.merge(d=-1 * min_requierd_bp).to_dataframe()
    bed.columns = ['Chromosome', 'Start', 'End'] + list(bed.columns[3:])
    return bed


def load_entire_hdf5(dct):
    if isinstance(dct, h5py.Dataset):
        return dct[()]
    ret = {}
    for k, v in dct.items():
        ret[k] = load_entire_hdf5(v)
    return ret

def dftopyranges(df):
    return pyranges.PyRanges(df)


def df2regionidentifier(df):
    return np.array(['%s:%d-%d' % (c, s, e) for c, s, e in
                                  zip(df['Chromosome'], df['Start'], df['End'])])


# This function will fetch the existing entries in f
# Make comparisons to the provided list
# Then return the uniq stuff in the provided list, existing stuff, and the new stuff
def Unify_meta_info(f,
                    addition_feats=[],
                    entries=['group', 'region'],
                    dtypes=['str', 'str']
                    ):
    results = []
    for feat, entry, dtype in zip(addition_feats, entries, dtypes):
        if entry in f:
            if dtype == 'str':
                existing = np.array(f[entry].asstr()[:])
            else:
                existing = np.array(f[entry][:])
        else:
            existing = np.array([])

        uniq = np.array(feat)
        new = uniq[~np.isin(uniq, existing)]
        results += [uniq, existing, new]

    return tuple(results)

def sort_region_identifier(regions_all):
    regions_all = pd.DataFrame([re.split(':|-', xx) for xx in regions_all], columns=['Chromosome', 'Start', 'End'])
    regions_all = dftopyranges(regions_all).df
    regions_all = ['%s:%d-%d' % (c, s, e) for c, s, e in
                   zip(regions_all['Chromosome'], regions_all['Start'], regions_all['End'])]
    return regions_all


# Load TF binding / habitation prediction model
def loadBindingScoreModel(h5Path):
    with h5py.File(h5Path, 'r') as h5file:
        footprintMean =np.array(h5file["footprint_mean"])
        footprintSd =np.array(h5file["footprint_sd"])
        weights = [np.array(h5file["weight_%d" % i]) for i in range(6)]
        weights = [torch.from_numpy(w).float() for w in weights]
    return_dict = {
        'footprintMean': torch.from_numpy(footprintMean).float(),
        'footprintSd': torch.from_numpy(footprintSd).float(),
        'scales': [10, 20, 30, 50, 80, 100],
        'weights': weights}

    return return_dict

# Load TF binding / habitation prediction model
def loadBindingScoreModel_pt(h5Path):
    model = torch.jit.load(h5Path)
    model.eval()
    return_dict = {
        'model': model,
        'scales': [int(xx) for xx in model.scales],
        'version': str(model.version)}

    return return_dict


def loadDispModel(h5Path):
    with h5py.File(h5Path, "r") as a:
        dispmodels = load_entire_hdf5(a)
        for model in dispmodels:
            dispmodels[model]['modelWeights'] = [torch.from_numpy(dispmodels[model]['modelWeights'][key]).float() for
                                                 key in ['ELT1', 'ELT2', 'ELT3', 'ELT4']]
    return dispmodels


def downloadDispModel():
    return None

def downloadTFBSModel():
    return None