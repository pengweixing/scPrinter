from __future__ import annotations

import os.path

from . import genome
from .utils import *
from .io import load_printer, PyPrinter
from scprinter.shift_detection import detect_shift
import pyBigWig
from tqdm.auto import tqdm, trange
from scipy.sparse import coo_matrix, hstack, vstack, csr_matrix, csc_matrix
from anndata import AnnData
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import uuid
import time
import snapatac2 as snap
import multiprocessing


def import_data(path, barcode, gff_file, chrom_sizes, extra_plus_shift,
                extra_minus_shift, tempname, **kwargs):
    if '.gz' not in path:
        if os.path.exists(path + '.gz'):
            print ("using gzipped file: %s.gz" % path)
            path = path + '.gz'
        else:
            print ("gzipping %s, because currently the backend requires gzipped file" % path)
            os.system('gzip %s' % path)
            path = path + '.gz'

    data = snap.pp.import_data(path,
                               file=tempname,
                               whitelist=barcode,
                               chrom_sizes=chrom_sizes,
                               shift_left=extra_plus_shift,
                               shift_right=extra_minus_shift,
                               **kwargs)
    data = frags_to_insertions(data, split=False)
    data.close()
    return tempname


def import_fragments(pathToFrags: str | list[str] | Path | list[Path],
                    barcodes: list[str] | list[list[str]] | Path | list[Path],
                    savename: str | Path,
                    genome: genome.Genome,
                    plus_shift: int = 4,
                    minus_shift: int = -5,
                    auto_detect_shift: bool = True,
                    unique_string: str | None = None,
                    **kwargs):

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
    genome: genome.Genome
        Genome object that contains all the necessary information for the genome
    plus_shift: int
        The shift **you have done** for the left end of the fragment. Default is 4,
        which is what the SHARE-seq pipeline does
    minus_shift: int
        The shift **you have done** for the right end of the fragment. Default is -5,
        which is what the SHARE-seq pipeline does
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

    if auto_detect_shift:
        print ("You are now using the beta auto_detect_shift function, this overwrites the plus_shift and minus_shift you provided")
        print ("If you believe the auto_detect_shift is wrong, please set auto_detect_shift=False")
        plus_shift, minus_shift = detect_shift(pathsToFrags[0], genome)
        print ("detected plus_shift and minus_shift are", plus_shift, minus_shift)

    # this is equivalent to do +4/-4 shift,
    # because python is 0-based, and I'll just use the right end as index
    # meaning that I'll do [end, end+1) as insertion position, not [end-1, end)
    extra_plus_shift = 4 - plus_shift
    extra_minus_shift = -5 - minus_shift
    # this function check is snapATAC2 fix the insertion in the future
    flag_ = check_snap_insertion()
    print ("snapatac2 shift check", flag_)
    if not flag_:
        flag_ = check_snap_insertion(0, 1)
        if not flag_:
            print ("raise an Issue please")
            raise EOFError
        else:
            extra_minus_shift += 1

    if 'low_memory' in kwargs:
        del  kwargs['low_memory']

    # this is a historical kwarg that snapatac2 takes, but not anymore
    if 'min_tsse' in kwargs:
        del kwargs['min_tsse']

    if len(pathsToFrags) == 1:
        path = pathsToFrags[0]
        if '.gz'not in path:
            if os.path.exists(path + '.gz'):
                print ("using gzipped file: %s.gz" % path)
                path = path + '.gz'
            else:
                print ("gzipping %s, because currently the backend requires gzipped file" % path)
                os.system('gzip %s' % path)
                path = path + '.gz'


        data = snap.pp.import_data(path,
                                   file=savename,
                                   whitelist=barcodes[0],
                                   # gene_anno=genome.fetch_gff(),
                                   chrom_sizes=genome.chrom_sizes,
                                   shift_left=extra_plus_shift,
                                   shift_right=extra_minus_shift,
                                   **kwargs)
        snap.metrics.tsse(data, genome.fetch_gff())
        frags_to_insertions(data, split=True)
    else:
        # with multiple fragments, store them in memory and concat
        # Should be able to support snapatac2.anndataset in the future, but, let's keep it this way for now
        adatas = []
        p_list = []
        pool = ProcessPoolExecutor(max_workers=20, mp_context=multiprocessing.get_context("spawn"))


        ct = 0
        bar = trange(len(pathsToFrags), desc="Importing fragments")
        for path, barcode in zip(pathsToFrags, barcodes):
            p_list.append(pool.submit(import_data, path, barcode,
                                      genome.fetch_gff(), genome.chrom_sizes,
                                      extra_plus_shift, extra_minus_shift,
                                      savename+"_part%d" %ct,**kwargs))


            # data = snap.pp.import_data(path,
            #                            file=savename+"_part%d" %ct,
            #                            whitelist=barcode,
            #                            # gene_anno=genome.fetch_gff(),
            #                            chrom_sizes=genome.chrom_sizes,
            #                            shift_left=extra_plus_shift,
            #                            shift_right=extra_minus_shift,
            #                            **kwargs)
            # frags_to_insertions(data)
            # data.close()
            # savepath = savename + "_part%d" % ct
            ct += 1
            bar.update(1)
            bar.refresh()
        for p in tqdm(as_completed(p_list), total=len(p_list)):
            savepath = p.result()
            adatas.append((savepath.split("_")[-1],savepath))
            sys.stdout.flush()

        data = snap.AnnDataSet(adatas=adatas, filename=savename+"_temp")

        data2 = snap.AnnData(filename=savename,
                             obs=data.obs[:])
        data2.obs_names = data.obs_names
        print ("start transferring insertions")
        start = time.time()
        insertion = data.adatas.obsm['insertion']

        indx = list(np.cumsum(data.uns['reference_sequences']['reference_seq_length']).astype('int'))
        start = [0] + indx
        end = indx
        for chrom, start, end in zip(
                data.uns['reference_sequences']['reference_seq_name'],
                start,
                end):
            data2.obsm['insertion_%s' % chrom] = insertion[:, start:end].tocsc()


        print ("takes", time.time() - start)
        data2.uns['reference_sequences'] = data.uns['reference_sequences']
        data.close()
        data2.close()

        for i in range(ct):
            os.remove(savename+"_part%d" %i)
        os.remove(savename+"_temp")
        data = snap.read(savename)

    data.uns['genome'] = f'{genome=}'.split('=')[0]
    data.uns['unique_string'] = unique_string
    data.close()

    return load_printer(savename, genome)

def make_peak_matrix(printer: PyPrinter,
                     regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
                     region_width: int | None = None,):
    """
    Generate a peak matrix for the given regions
    Parameters
    ----------
    printer
    regions

    Returns
    -------

    """
    regions = regionparser(regions, printer, region_width)
    region_identifiers = df2regionidentifier(regions)

    insertion_profile = printer.fetch_insertion_profile()
    res = []
    for i, (chrom, start, end) in enumerate(zip(tqdm(regions.iloc[:, 0], desc="Making peak matrix"), regions.iloc[:, 1], regions.iloc[:, 2])):
        v = csc_matrix(insertion_profile[chrom][:, start:end].sum(axis=-1), dtype='uint16')
        res.append(v)
    res = hstack(res).tocsr()
    print (res.shape)
    adata = AnnData(X=res)
    adata.obs.index = printer.insertion_file.obs_names[:]
    adata.var.index = region_identifiers
    return adata



def collapse_barcodes(*args, **kwargs):
    sync_footprints(*args, **kwargs)

def sync_footprints(printer: PyPrinter,
                    cell_grouping: list[list[str]] | list[str] | np.ndarray,
                    group_names: list[str] | str,):
    """
    Generate bigwig files for each group which can be used for synchronized footprint visualization

    Parameters
    ----------
    printer: PyPrinter object
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

    cell_grouping = cell_grouping2cell_grouping_idx(printer,
                                                    cell_grouping)
    insertion_profile = printer.fetch_insertion_profile()
    chrom_list = list(insertion_profile.keys())
    chunksize = 1000000

    a = printer.insertion_file.uns['group_bigwig'] if 'group_bigwig' in printer.insertion_file.uns else {}
    a['bias'] = printer.insertion_file.uns['bias_bw']

    for name, grouping in zip(group_names, cell_grouping):
        print ("Creating bigwig for %s" % name)

        path = os.path.join(os.path.dirname(printer.file_path),
                            '%s.bw' % name)

        bw = pyBigWig.open(path, 'w')
        header = []
        for chrom in chrom_list:
            sig = insertion_profile[chrom]
            length = sig.shape[-1]
            header.append((chrom, length))
        bw.addHeader(header, maxZooms=10)
        for chrom in tqdm(chrom_list):
            sig = insertion_profile[chrom]
            for i in range(0, sig.shape[-1], chunksize):
                temp_sig = sig[:, slice(i, i+chunksize)]
                if temp_sig.nnz == 0:
                    continue
                pseudo_bulk = coo_matrix(temp_sig[grouping].sum(axis=0))
                if len(pseudo_bulk.data) == 0:
                    continue

                col, data = pseudo_bulk.col, pseudo_bulk.data
                indx = np.argsort(col)

                bw.addEntries(str(chrom),
                    col[indx] + i,
                    values=data[indx].astype('float'), span=1, )
        bw.close()
        a[str(name)] = str(path)

    printer.insertion_file.uns['group_bigwig'] = a

def seq_model_config(printer: PyPrinter,
                     peak_file: str | Path,
                     cell_grouping: list[list[str]] | list[str] | np.ndarray,
                     group_names: list[str] | str,
                     overwrite_bigwig=True,
                     model_name=None,
                     model_configs={}):
    if type(group_names) not in [np.ndarray, list]:
        group_names = [group_names]
        cell_grouping = [cell_grouping]
    if len(group_names) > 1:
        raise NotImplementedError("Currently only support one group at a time")
    if model_name is None and 'savename' not in model_configs:
        raise ValueError("Please provide a model name or a savename in model_configs")
    if 'group_bigwig' not in printer.insertion_file.uns:
        printer.insertion_file.uns['group_bigwig'] = {}

    for name, grouping in zip(group_names, cell_grouping):
        if name in printer.insertion_file.uns['group_bigwig']:
            if not overwrite_bigwig:
                print ("bigwig for %s already exists, skip" % name)
                continue
        sync_footprints(printer, grouping, name)

    template_json = {
      "peaks": peak_file,
      "signals": [printer.insertion_file.uns['group_bigwig'][name] for name in group_names],
      "bias": printer.insertion_file.uns['bias_bw'],
      "split":{
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
        ]},
      "max_jitter": 128,
      "reverse_compliment": True,
      "n_filters": 768,
      "bottleneck_factor": 0.5,
      "amp": True,
      "ema": True,
      "groups": 1,
      "n_inception_layers": 8,
      "n_layers": 8,
      "inception_layers_after": True,
      "activation": "gelu",
      "batch_norm_momentum": 0.1,
      "depthwise_separable": False,
      "activation_in_between": False,
      "dilation_base": 1,
      "rezero": False,
      "batch_norm": True,
      "batch_size": 64,
      "head_kernel_size": 1,
      "kernel_size": 3,
      "weight_decay": 1e-3,
      "lr": 1e-3,
      "scheduler": False,
      "savename": model_name,
      "replicate": 1,
      "coverage_weight": 0.1,
      "no_inception": False,
      "inception_version": 2
    }
    for key in model_configs:
        template_json[key] = model_configs[key]

    return template_json




