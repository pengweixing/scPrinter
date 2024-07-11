from __future__ import annotations

import multiprocessing
import os.path
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
import pyBigWig
import snapatac2 as snap
from anndata import AnnData
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, hstack, vstack
from tqdm.auto import tqdm, trange

from .genome import Genome
from .io import load_printer, scPrinter
from .shift_detection import detect_shift
from .utils import *


def import_data(
    path,
    barcode,
    genome,
    plus_shift,
    minus_shift,
    auto_detect_shift,
    tempname,
    close_after_import=True,
    **kwargs,
):
    """
    Imports data from a specified path, applies whitelist,
    and shifts the fragments if necessary.

    Parameters:
    path (str): The path to the input fragments file.
    barcode (list[str]): The whitelist of barcodes to be included in the data.
    gff_file (str): The path to the GFF file containing gene annotations.
    chrom_sizes (dict[str, int]): A dictionary mapping chromosome names to their sizes.
    extra_plus_shift (int): The extra shift to be added to the plus strand fragments.
    extra_minus_shift (int): The extra shift to be added to the minus strand fragments.
    tempname (str): The temporary name for the output file.
    kwargs: Additional keyword arguments to be passed to the import_data function.

    Returns:
    str: The temporary name of the output file.
    """

    if ".gz" not in path:
        if os.path.exists(path + ".gz"):
            print("using gzipped file: %s.gz" % path)
            path = path + ".gz"
        else:
            print("gzipping %s, because currently the backend requires gzipped file" % path)
            os.system("gzip %s" % path)
            path = path + ".gz"
    if auto_detect_shift:
        print(
            "You are now using the beta auto_detect_shift function, this overwrites the plus_shift and minus_shift you provided"
        )
        print("If you believe the auto_detect_shift is wrong, please set auto_detect_shift=False")
        plus_shift, minus_shift = detect_shift(path, genome)
        print("detected plus_shift and minus_shift are", plus_shift, minus_shift, "for", path)

    # this is equivalent to do +4/-4 shift,
    # because python is 0-based, and I'll just use the right end as index
    # meaning that I'll do [end, end+1) as insertion position, not [end-1, end)
    extra_plus_shift = 4 - plus_shift
    extra_minus_shift = -5 - minus_shift

    # Import data from the fragments file.
    # shift_left and shift_right are set all to zeros now, the extra shifting now happens during the frags to insertion part.
    data = snap.pp.import_data(
        path,
        file=tempname,
        whitelist=barcode,
        chrom_sizes=genome.chrom_sizes,
        shift_left=0,
        shift_right=0,
        **kwargs,
    )
    data = frags_to_insertions(
        data, split=True, extra_plus_shift=extra_plus_shift, extra_minus_shift=extra_minus_shift
    )
    if close_after_import:
        data.close()
        return tempname
    else:
        return data


def import_fragments(
    pathToFrags: str | list[str] | Path | list[Path],
    barcodes: list[str] | list[list[str]] | Path | list[Path],
    savename: str | Path,
    genome: Genome,
    plus_shift: int = 4,
    minus_shift: int = -5,
    auto_detect_shift: bool = True,
    unique_string: str | None = None,
    n_jobs: int = 20,
    **kwargs,
):
    """
    Import ATAC fragments into single cell genomewide insertion profile backed in anndata format

    Parameters
    ----------
    pathToFrags: str | list[str] | Path | list[Path]
        Path or List of paths to the fragment files. When multiple files are provided,
        they will be separately imported and concatenated.
    barcodes: list[str]
        List of barcodes to be whitelisted. If None, all barcodes will be used. Recommend to use.
        If you input one fragments file, the barcodes can either be a list of barcodes, or the path to a barcode file
        where each line is one barcode to include.
        If you input multiple fragments files, the barcodes should be a list of list of barcodes, or a list of paths.
    savename: str | Path
        Path to save the anndata object
    genome: Genome
        Genome object that contains all the necessary information for the genome
    plus_shift: int
        The shift **you have done** for the left end of the fragment. Default is 4,
        which is what the standard pipeline does
    minus_shift: int
        The shift **you have done** for the right end of the fragment. Default is -5,
        which is what the standard pipeline does (terra SHARE-seq pipeline now does -4)
    auto_detect_shift: bool
        Whether to automatically detect the shift. Default is True. It will overwrite the plus_shift and minus_shift you provided.
    unique_string: str
        A unique string. If None, a random UUID will be used. It is used to create a unique name for global variables.
        Most of the time you don't need to set it.
    n_jobs: int
        Number of workers for parallel processing. Default is 20. Only used when multiple fragments files are provided.
    kwargs
        Other arguments passed to snapatac2.pp.import_data

    Returns
    -------

    """
    if unique_string is None:
        unique_string = str(uuid.uuid4())
    # hoping for a list, but if it's a string, treated it as a list
    if type(pathToFrags) is str:
        pathsToFrags = [pathToFrags]
        barcodes = [barcodes]

    else:
        pathsToFrags = pathToFrags

    # No longer needed since now snapATAC2 won't handle the frags -> insertions
    # # this function check is snapATAC2 fix the insertion in the future
    # flag_ = check_snap_insertion()
    # print("snapatac2 shift check", flag_)
    # if not flag_:
    #     flag_ = check_snap_insertion(0, 1)
    #     if not flag_:
    #         print("raise an Issue please")
    #         raise EOFError
    #     else:
    #         extra_minus_shift += 1

    # these are historical_kwargs that snapatac2 takes, but not anymore
    for historical_kwarg in ["min_tsse", "low_memory"]:
        if historical_kwarg in kwargs:
            del kwargs[historical_kwarg]

    if len(pathsToFrags) == 1:
        path = pathsToFrags[0]
        data = import_data(
            path,
            barcodes[0],
            genome,
            plus_shift,
            minus_shift,
            auto_detect_shift,
            savename,
            close_after_import=False,
            **kwargs,
        )
    else:
        # with multiple fragments, store them in memory and concat
        # Should be able to support snapatac2.anndataset in the future, but, let's keep it this way for now
        adatas = []
        p_list = []
        pool = ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn")
        )

        ct = 0
        bar = trange(len(pathsToFrags), desc="Importing fragments")
        for path, barcode in zip(pathsToFrags, barcodes):
            p_list.append(
                pool.submit(
                    import_data,
                    path,
                    barcode,
                    genome,
                    plus_shift,
                    minus_shift,
                    auto_detect_shift,
                    savename + "_part%d" % ct,
                    True,
                    **kwargs,
                )
            )
            ct += 1
            bar.update(1)
            bar.refresh()
        for p in tqdm(as_completed(p_list), total=len(p_list)):
            savepath = p.result()
            adatas.append((savepath.split("_")[-1], savepath))
            sys.stdout.flush()

        data = snap.AnnDataSet(adatas=adatas, filename=savename + "_temp")

        # Now transfer the insertions to the final AnnData object
        data2 = snap.AnnData(filename=savename)
        data2.obs = data.obs[:]
        data2.obs_names = data.obs_names
        print("start transferring insertions")
        for chrom in data.uns["reference_sequences"]["reference_seq_name"]:
            key = f"insertions_{chrom}"
            insertions = data.uns[key]
            data2.obsm[key] = insertions.tocsc()

        data2.uns["reference_sequences"] = data.uns["reference_sequences"]
        data.close()
        data2.close()

        for i in range(ct):
            os.remove(savename + "_part%d" % i)
        os.remove(savename + "_temp")
        data = snap.read(savename)

    data.uns["genome"] = f"{genome=}".split("=")[0]
    data.uns["unique_string"] = unique_string
    snap.metrics.tsse(data, genome.fetch_gff())
    data.close()

    return load_printer(savename, genome)


def mean_norm_counts(adata):
    """
    Normalize the count matrix by dividing each cell by its mean count across all peaks.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the count matrix to be normalized.

    Returns
    -------
    The normalization is performed in place, modifying the input AnnData object.
    """
    xx = csr_matrix(adata.X / adata.X.mean(axis=1))
    adata.X = xx
    return adata


def make_gene_matrix(
    printer: scPrinter,
    strategy: Literal["gene", "promoter", "tss"] = "gene",
    cell_grouping: list[list[str]] | list[str] | np.ndarray | None = None,
    group_names: list[str] | str | None = None,
    sparse=True,
):
    """
    Generate a gene matrix based on the given strategy.

    Parameters
    ----------
    printer : scPrinter
        The scPrinter object containing the insertion profile and other relevant information.
    strategy : Literal["gene", "promoter", "tss"], optional
        The strategy to generate the gene matrix. Default is "gene".
    cell_grouping : list[list[str]] | list[str] | np.ndarray | None, optional
        The cell grouping for creating the gene matrix. Default is None.
    group_names : list[str] | str | None, optional
        The names of the groups. Default is None.
    sparse : bool, optional
        Whether to use sparse matrices for storing the gene matrix. Default is True.

    Returns
    -------
    adata : AnnData
        The gene matrix in AnnData format.
    """

    if strategy == "gene":
        # Fetch all genes from the database
        db = printer.gff_db
        info = []
        for gene in db.features_of_type("gene"):
            if gene.chrom not in printer.genome.chrom_sizes:
                continue
            info.append([gene.chrom, gene.start, gene.end, gene.id])
        regions = pd.DataFrame(info, columns=["chrom", "start", "end", "gene_id"])
        adata = make_peak_matrix(
            printer, regions, cell_grouping=cell_grouping, group_names=group_names, sparse=sparse
        )

        adata.var.index = regions["gene_id"]
        return adata
    else:
        print("Sorry, the current implementation only supports 'gene' strategy.")


def make_peak_matrix(
    printer: scPrinter,
    regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
    region_width: int | None = None,
    cell_grouping: list[list[str]] | list[str] | np.ndarray | None = None,
    group_names: list[str] | str | None = None,
    sparse=True,
):
    """
    Generate a peak matrix for the given regions.

    Parameters
    ----------
    printer : scPrinter
        The scPrinter object containing the insertion profile and other relevant information.
    regions : str | Path | pd.DataFrame | pyranges.PyRanges | list[str]
        The regions for which the peak matrix needs to be generated. It can be a file path, a DataFrame,
        a PyRanges object, or a list of region identifiers.
    region_width : int | None, optional
        The width of the regions. If None, the width will be determined based on the input regions.
    cell_grouping : list[list[str]] | list[str] | np.ndarray | None, optional
        The cell grouping for creating the peak matrix. If None, the peak matrix will be created at single
        cell resolution.
    group_names : list[str] | str | None, optional
        The names of the groups. If None, the group names will be determined based on the cell grouping.
    sparse : bool, optional
        Whether to use sparse matrices for storing the peak matrix. Default is True.

    Returns
    -------
    adata : AnnData
        The peak matrix in AnnData format.
    """

    # Parse the regions to a bed DataFrame
    regions = regionparser(regions, printer, region_width)
    region_identifiers = df2regionidentifier(regions)

    # Get the insertion profiles
    insertion_profile = printer.fetch_insertion_profile()
    res = []

    # depending on whether to use sparse matrices, we use different sparse matrix classes
    # COO first, because it makes vstack more efficient
    # Then CSC later, because it makes hstack more efficient
    wrapper1 = coo_matrix if sparse else np.array
    wrapper2 = csc_matrix if sparse else np.array

    if cell_grouping is not None:
        cell_grouping_idx = cell_grouping2cell_grouping_idx(printer, cell_grouping)
        if type(group_names) not in [np.ndarray, list, None]:
            group_names = [group_names]
            cell_grouping = [cell_grouping]

    for i, (chrom, start, end) in enumerate(
        zip(
            tqdm(regions.iloc[:, 0], desc="Making peak matrix"),
            regions.iloc[:, 1],
            regions.iloc[:, 2],
        )
    ):
        # First create single cell resolution -> CSC matrix for each region
        v = wrapper2(insertion_profile[chrom][:, start:end].sum(axis=-1), dtype="uint16")
        res.append(v)
    # We now have a cell x region peak matrix
    res = hstack(res).tocsr() if sparse else np.hstack(res)
    # Now merge them into pseudobulk x region AnnData object
    if cell_grouping is not None:
        a = [wrapper1(np.array(res[barcodes].sum(axis=0))) for barcodes in cell_grouping_idx]
        res = wrapper2(vstack(a))

    adata = AnnData(X=res)
    adata.obs.index = printer.insertion_file.obs_names[:] if cell_grouping is None else group_names
    adata.var.index = region_identifiers
    return adata


"""
These are both historical names for the same function. My bad...
"""


def collapse_barcodes(*args, **kwargs):
    sync_footprints(*args, **kwargs)


def sync_footprints(*args, **kwargs):
    export_bigwigs(*args, **kwargs)


def export_bigwigs(
    printer: scPrinter,
    cell_grouping: list[list[str]] | list[str] | np.ndarray,
    group_names: list[str] | str,
):
    """
    Generate bigwig files for each group which can be used for synchronized footprint visualization

    Parameters
    ----------
    printer: scPrinter object
        The printer object you generated by `scprinter.pp.import_fragments` or loaded by `scprinter.load_printer`
    cell_grouping: list[list[str]] | list[str] | np.ndarray
        The cell grouping you want to visualize, specifiec by a list of the cell barcodes belong to this group, e.g.
        `['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112', 'ACAGTGGT,ACAGTGGT,ATCACGTT,BUENSS112', 'ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']`.  If you want to visualize multiple groups, you can provide a list of lists, e.g.
        `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`.
    group_names: list[str] | str
        The name of the group you want to visualize.
        If you want to visualize multiple groups, you can provide a list of names, e.g. `['group1', 'group2']`

    Returns
    -------

    """
    if type(group_names) not in [np.ndarray, list]:
        group_names = [group_names]
        cell_grouping = [cell_grouping]

    cell_grouping = cell_grouping2cell_grouping_idx(printer, cell_grouping)
    insertion_profile = printer.fetch_insertion_profile()
    chrom_list = list(insertion_profile.keys())
    chunksize = 1000000

    a = (
        printer.insertion_file.uns["group_bigwig"]
        if "group_bigwig" in printer.insertion_file.uns
        else {}
    )
    a["bias"] = printer.insertion_file.uns["bias_bw"]

    for name, grouping in zip(group_names, cell_grouping):
        print("Creating bigwig for %s" % name)

        path = os.path.join(os.path.dirname(printer.file_path), "%s.bw" % name)

        bw = pyBigWig.open(path, "w")
        header = []
        for chrom in chrom_list:
            sig = insertion_profile[chrom]
            length = sig.shape[-1]
            header.append((chrom, length))
        bw.addHeader(header, maxZooms=10)
        for chrom in tqdm(chrom_list):
            sig = insertion_profile[chrom]
            for i in range(0, sig.shape[-1], chunksize):
                temp_sig = sig[:, slice(i, i + chunksize)]
                if temp_sig.nnz == 0:
                    continue
                pseudo_bulk = coo_matrix(temp_sig[grouping].sum(axis=0))
                if len(pseudo_bulk.data) == 0:
                    continue

                col, data = pseudo_bulk.col, pseudo_bulk.data
                indx = np.argsort(col)

                bw.addEntries(
                    str(chrom),
                    col[indx] + i,
                    values=data[indx].astype("float"),
                    span=1,
                )
        bw.close()
        a[str(name)] = str(path)

    printer.insertion_file.uns["group_bigwig"] = a
