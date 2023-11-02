from __future__ import annotations
from . import genome
from .utils import *
from .io import load_printer, PyPrinter
import pyBigWig
from tqdm.auto import tqdm
from scipy.sparse import coo_matrix
import numpy as np
import anndata
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import uuid
import time
def import_data(path, barcode, gff_file, chrom_sizes, extra_plus_shift,
                extra_minus_shift, tempname, **kwargs):
    data = snap.pp.import_data(path,
                               file=tempname,
                               whitelist=barcode,
                               chrom_sizes=chrom_sizes,
                               shift_left=extra_plus_shift,
                               shift_right=extra_minus_shift,
                               **kwargs)
    data = frags_to_insertions(data)
    data.close()
    return tempname


def import_fragments(pathToFrags: str | list[str] | Path | list[Path],
                    barcodes: list[str] | list[list[str]] | Path | list[Path],
                    savename: str | Path,
                    genome: genome.Genome,
                    plus_shift: int = 4,
                    minus_shift: int = -5,
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
        data = snap.pp.import_data(pathsToFrags[0],
                                   file=savename,
                                   whitelist=barcodes[0],
                                   # gene_anno=genome.fetch_gff(),
                                   chrom_sizes=genome.chrom_sizes,
                                   shift_left=extra_plus_shift,
                                   shift_right=extra_minus_shift,
                                   **kwargs)
        frags_to_insertions(data)
    else:
        # with multiple fragments, store them in memory and concat
        # Should be able to support snapatac2.anndataset in the future, but, let's keep it this way for now
        adatas = []
        p_list = []
        pool = ProcessPoolExecutor(max_workers=20)


        ct = 0
        for path, barcode in zip(pathsToFrags, barcodes):
            p_list.append(pool.submit(import_data, path, barcode,
                                      genome.fetch_gff(), genome.chrom_sizes,
                                      extra_plus_shift, extra_minus_shift,
                                      savename+"_part%d" %ct,**kwargs))
            ct += 1
            # data = snap.pp.import_data(path,
            #                            file=None,
            #                            whitelist=barcode,
            #                            gff_file=genome.fetch_gff(),
            #                            chrom_size=genome.chrom_sizes,
            #                            shift_left=extra_plus_shift,
            #                            shift_right=extra_minus_shift,
            #                                        ** kwargs)
        for p in tqdm(as_completed(p_list), total=len(p_list)):
            path = p.result()
            adatas.append((path.split("_")[-1],path))

        data = snap.AnnDataSet(adatas=adatas, filename=savename+"_temp")

        data2 = snap.AnnData(filename=savename,
                             obs=data.obs[:])
        data2.obs_names = data.obs_names
        print ("start transferring insertions")
        start = time.time()
        data2.obsm['insertion'] = data.adatas.obsm['insertion']
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
        bw.addHeader(header, maxZooms=0)
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


