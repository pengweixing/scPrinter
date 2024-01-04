from __future__ import annotations

import gc
import sys

import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["NUMEXPR_MAX_THREADS"] = '1'
from . import motifs
from . import getTFBS
from . import getBias
from . import getFootprint
from .io import PyPrinter
from .fetch import _get_group_atac, _get_group_atac_bw
from .utils import *
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import anndata
import itertools
import warnings
import math
import torch
from tqdm.auto import trange, tqdm
import snapatac2 as snap
from scipy.sparse import csr_matrix, vstack, csc_matrix
import pandas as pd
import pyranges
from pathlib import Path
from sklearn.metrics import pairwise_distances
import time

def set_global_disp_model(printer):
    globals()[printer.unique_string + "_dispModels"] = printer.dispersionModel

def set_global_insertion_profile(printer):
    globals()[printer.unique_string + "insertion_profile"] = printer.fetch_insertion_profile()

class BackedBindingScoreAnnData():
    """
    A wrapper class for snap.AnnData to allow for lazy loading of sites
    """
    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        args
            args that will be passed to snap.AnnData
        kwargs
            kwargs that will be passed to snap.AnnData
        """
        self.adata = snap.AnnData(*args, **kwargs)
    def __getattr__(self, attr):
        return getattr(self.adata, attr)
    def __repr__(self):
        return self.adata.__repr__()
    def fetch_site(self, identifier):
        """
        Fetch the site information from the underlying hdf5 file
        Parameters
        ----------
        identifier: str
            the identifier of the footprint region, e.g. "chr1:1294726-1295726"

        Returns
        -------
        pd.DataFrame
            a dataframe containing the site information

        """
        site_index = pd.read_hdf(str(self.uns['sites']), 'site_id').loc[identifier]

        return pd.read_hdf(str(self.uns['sites']),
                           'chunk_%d' % site_index['chunk']).iloc[site_index['start']:site_index['end']]

class BindingScoreAnnData():
    """
    A wrapper class for anndata.AnnData to allow for lazy loading of sites
    """
    def __init__(self, *args, **kwargs):
        self.adata = anndata.AnnData(*args, **kwargs)
    def __getattr__(self, attr):
        return getattr(self.adata, attr)
    def __repr__(self):
        return self.adata.__repr__()
    def fetch_site(self, identifier):
        """
        Fetch the site information from the underlying hdf5 file
        Parameters
        ----------
        identifier: str
            the identifier of the footprint region, e.g. "chr1:1294726-1295726"

        Returns
        -------
        pd.DataFrame
            a dataframe containing the site information
        """
        site_index = pd.read_hdf(str(self.uns['sites']), 'site_id').loc[identifier]

        return pd.read_hdf(str(self.uns['sites']),
                           'chunk_%d' % site_index['chunk']).iloc[site_index['start']:site_index['end']]


'''
underlying function for calculaing batch of binding scores
'''
def _get_binding_score_batch(
        region_identifiers,
        cell_grouping,
        Tn5Bias,  # Numeric vector of predicted Tn5 bias for the current region
        regions,  # pyRanges object for the current region
        sites=None,  # pyRanges object. Genomic ranges of motif matches in the current region
        # Must have $score attribute which indicates how well the motif is matched (between 0 and 1)
        # If NULL, the region will be divided into equally sized tiles for TF binding scoring
        tileSize=10,  # Size of tiles if sites is NULL
        contextRadius=100,
        unique_string=None,
        downsample=1,
        strand='*'):

    result = []
    # Create a fake pyprinter object to pass on
    # Why we need a fake one?
    # Because these functions are created to be compatible with PyPrinter centered objects
    # But it would be slow to pass the real printer project, because python would pickle everything
    insertion_profile = globals()[unique_string+'insertion_profile']
    dispModels = globals()[unique_string + "_dispModels"]
    bindingscoremodel = globals()[unique_string + "_bindingscoremodel"]
    printer = PyPrinter(insertion_profile=insertion_profile)
    for i, region_identifier in enumerate(region_identifiers):
        # assumes the cell_grouping
        if sites is None:
            site = None
        else:
            site = sites[i]

        atac = _get_group_atac(printer, cell_grouping, regions.iloc[i])
        if downsample < 1:
            n_remove_insertions = int(np.sum(atac) * (1 - downsample))
            v = atac.data
            reduce = np.random.choice(len(v), n_remove_insertions, replace=True, p=v / np.sum(v))
            reduce, reduce_ct = np.unique(reduce, return_counts=True)
            v[reduce] -= reduce_ct
            v[v < 0] = 0

        result.append(getTFBS.getRegionBindingScore(
            region_identifier,
            atac,
            Tn5Bias[i,],
            dftopyranges(regions.iloc[i:i + 1]),
            dispModels,
            bindingscoremodel,
            site,
            tileSize,
            contextRadius,
            strand))

        del atac
    return result

def get_binding_score(printer: PyPrinter,
                    cell_grouping: list[str] | list[list[str]] | np.ndarray,
                    group_names: list[str] | str,
                    regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
                    region_width: int | None = None,
                    model_key: str ='TF',
                    nCores: int = 16,
                    contextRadius: int = 100,
                    save_key: str = None,
                    backed: bool = True,
                    overwrite: bool = False,
                    motif_scanner: motifs.Motifs | None = None,
                    manual_sites: list[pd.DataFrame] | None = None,
                    tile:int=10,
                    open_anndata=True,
                    downsample=1,
                    strand="*",
                    cache_factor=2,):

    """
    Calculate the binding score for a given region and cell group, stores the result in the anndata object.
    The resulting anndata object will be linked to the printer object with the key 'save_key'.

    Parameters
    ----------
    printer: PyPrinter
        The printer object you generated by `scprinter.pp.import_fragments` or loaded by `scprinter.load_printer`
    cell_grouping: list[list[str]] | list[str] | np.ndarray
        The cell grouping you want to visualize, specifiec by a list of the cell barcodes belong to this group, e.g.
        `['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112', 'ACAGTGGT,ACAGTGGT,ATCACGTT,BUENSS112', 'ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']`.  If you want to visualize multiple groups, you can provide a list of lists, e.g.
        `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`.
    group_names: list[str] | str
        The name of the group you want to visualize.
        If you want to visualize multiple groups, you can provide a list of names, e.g. `['group1', 'group2']`
    regions: str | pd.DataFrame | pyranges.PyRanges | list[str]
        The genomic regions you want to calculate the binding score.
        You can provide a string of the path to the region bed file,
        a pandas dataframe with the first three columns correspond to [chrom, start, end],
        a pyranges object of the regions,
        or a list of region identifiers, e.g. `['chr1:1000-2000', 'chr1:3000-4000']`
        or a list of gene identifiers, e.g. `['Gene:CTCF', 'Gene:GATA1']`,
        the transcript start site will be used as the region center
    region_width: int | None
        The width of the region.
        The regions you provided would be resized to this width with fixed center point of each region
    model_key: str
        The key of the model you want to use for the binding score calculation, which you specified when loading the model.
    nCores: int
        Number of parallel process to use
    contextRadius: int
        The radius of the context region used for the binding score calculation, default 100
    save_key: str
        The name of the anndata object you want to save the binding score result to.
    backed: bool
        Whether to save the result to a backed anndata object.
    overwrite: bool
        Whether to overwrite if there is an existing result.
    motif_scanner: motifs.Motifs | None
        The motif scanner you want to use for the binding score calculation. When passing one, the binding score will be
        calculated only on sites that have a motif matching. When passing None,
        the binding score will be calculated on tiled regions with step size 10

    Returns
    -------
    AnnData
        This AnnData can be accessed through `printer.bindingscoreadata[save_key]`
        The Anndata object stores `n_obs x n_var` matrix, where `n_obs` is the number of groups, `n_vars` is the number of regions.
        The `obs` stores the group associated properties, the `var` stores the region associated properties.
        The `obsm` stores the binding score matrix, with the key of region_identifiers such as `chr1:1000-2000`.
        The value of each `obsm` has the shape of `n_obs x n_sites`, n_sites is the number of sites in the region.
        The `uns` stores the model parameters.

    """

    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        if type(group_names) not in [np.ndarray, list]:
            group_names = [group_names]
            cell_grouping = [cell_grouping]

        regions = regionparser(regions, printer, region_width)
        region_identifiers = df2regionidentifier(regions)

        if 'binding score' not in printer.uns:
            printer.uns['binding score'] = {}

        if save_key is None:
            save_key = "%s_BindingScore" % model_key

        save_path = os.path.join(os.path.dirname(printer.file_path),
                                 "%s.h5ad" % save_key) if backed else 'None'


        if save_key in printer.uns['binding score'] or os.path.exists(save_path):
            # Check duplicates...
            if not overwrite:
                print("detected %s, not allowing overwrite" % save_key)
                return
            else:
                try: os.remove(os.path.join(os.path.dirname(printer.file_path),
                             "%s_sites.h5ad" % save_key))
                except: pass
                try: os.remove(save_path)
                except: pass

        # results will be saved in a new key
        print("Creating %s in printer.bindingscoreadata" % save_key)
        print("obs=groups, var=regions")

        df_obs = pd.DataFrame({'name': group_names, 'id': np.arange(len(group_names))}, index = group_names)
        df_var = regions
        df_var['identifier'] = region_identifiers
        df_var.index = region_identifiers

        if backed:
            adata = BackedBindingScoreAnnData(filename=save_path, X=csr_matrix((len(group_names),
                                                                   len(regions))),
                                 obs=df_obs, var=df_var)

            adata.adata.obs_names = [str(xx) for xx in group_names]
            adata.adata.var_names = region_identifiers
            adata.uns['sites'] = str(os.path.join(os.path.dirname(printer.file_path),
                                 "%s_sites.h5ad" % save_key))

        else:
            adata = BindingScoreAnnData(X=csr_matrix((len(group_names),
                                                  len(regions))),
                                    obs=df_obs, var=df_var)
            adata.uns['sites'] = str(os.path.join(os.path.dirname(printer.file_path),
                                                  "%s_sites.h5ad" % save_key))


        a = printer.uns['binding score']
        a[save_key] = str(save_path)
        printer.uns['binding score'] = a
        assert model_key in printer.bindingScoreModel, "%s not loaded, run printer.load_bindingscore_model first" % model_key
        assert printer.dispersionModel is not None, "dispersion Model not loaded, run printer.load_disp_model first"



        dispersionModels = printer.dispersionModel
        bindingscoremodel = printer.bindingScoreModel[model_key]

        scales = bindingscoremodel['scales']
        model_version = bindingscoremodel['version']
        # Add model version to the anndata
        adata.uns['version'] = model_version
        # Only keep the related scales to avoid sending unnecessary stuff back and forth
        dispModels = [dispersionModels[str(scale)]
                      for scale in scales]

        # global insertion_profile, dispModels, bindingscoremodel
        unique_string = printer.unique_string
        globals()[unique_string + "insertion_profile"] = printer.fetch_insertion_profile()
        globals()[unique_string + "_dispModels"] = dispModels
        globals()[unique_string + "_bindingscoremodel"] = printer.bindingScoreModel[model_key]

        cell_grouping = cell_grouping2cell_grouping_idx(printer,
                                                        cell_grouping)
        print ("get bias")
        seqBias = getBias.getPrecomputedBias(printer.insertion_file.uns['bias_path'],
                                             regions,
                                             savePath=None)
        print ("finishing bias")
        small_chunk_size = min(int(math.ceil(len(regions) / nCores) + 1), 100)

        pool = ProcessPoolExecutor(max_workers=nCores)
        p_list = []
        sites = None
        width = regions.iloc[0]['End'] - regions.iloc[0]['Start']

        chunk_ct = 0
        index_df = []

        bar1 = trange(0, len(regions), small_chunk_size, desc='submitting jobs')
        bar = trange(len(region_identifiers), desc='collecting Binding prediction')
        time = 0
        if backed:
            adata.close()
            adata = h5py.File(save_path, 'r+', locking=False)
            adata_obsm = adata['obsm']
        for i in bar1:
            if motif_scanner is not None:
                sites = motif_scanner.scan_motif(regions.iloc[i:i+small_chunk_size,], clean=False, concat=False)
                new_sites = []
                for j, site in enumerate(sites):
                    if len(site) == 0:
                        new_sites.append(site)
                        continue
                    # [bed_coord['chr'], bed_coord['start'],
                    #   bed_coord['end'], bed_coord['index'],
                    #   name, score,
                    #   strand, motif_start, motif_end
                    site = pd.DataFrame({'Chromosome': site[:, 0],
                                         'Start': site[:, 1].astype('int') + site[:, -2].astype('int'),
                                         'End': site[:, 1].astype('int') + site[:, -1].astype('int'),
                                         'Strand': site[:, -3],
                                         'TF': site[:, 4], 'Score': site[:, 5].astype('float')})


                    relativePos = (np.array(site['Start'] + site['End']) * 0.5 - int(regions['Start'].iloc[i+j])).astype(
                        'int')
                    # Only keep sites with distance to CRE edge >= contextRadius
                    siteFilter = (relativePos > contextRadius) & (relativePos <= (width - contextRadius))
                    site = site.iloc[siteFilter]
                    new_sites.append(site)

                sites = new_sites
                if any([len(xx) > 0 for xx in sites]):
                    pd.concat([xx for xx in sites if len(xx) > 0], axis=0).to_hdf(os.path.join(os.path.dirname(printer.file_path),
                                             "%s_sites.h5ad" % save_key), 'chunk_%d' %  chunk_ct, complevel=4)
                    site_index = np.cumsum([0] + [len(xx) for xx in sites])
                    index_df.append(pd.DataFrame({'start':site_index[:-1],
                                             'end': site_index[1:],
                                             'chunk': [chunk_ct] * (len(site_index)-1)},
                                            index=region_identifiers[i:i+small_chunk_size]))


                    chunk_ct += 1

            if manual_sites is not None:
                sites = manual_sites[i:i+small_chunk_size]
            p_list.append(pool.submit(
                _get_binding_score_batch,
                region_identifiers[i:i+small_chunk_size],
                cell_grouping,
                seqBias[i:i+small_chunk_size,],  # Numeric vector of predicted Tn5 bias for the current region
                regions.iloc[i:i+small_chunk_size,],  # pyRanges object for the current region
                sites,  # pyRanges object. Genomic ranges of motif matches in the current region
                # Must have $score attribute which indicates how well the motif is matched (between 0 and 1)
                # If NULL, the region will be divided into equally sized tiles for TF binding scoring
                tile,  # Size of tiles if sites is NULL
                contextRadius,
                printer.unique_string,
                downsample,
                strand
            ))

            if len(p_list) > nCores * cache_factor:
                for p in as_completed(p_list):
                    rs = p.result()
                    for r in rs:
                        if 'BindingScore' not in r:
                            print (r, "error?")
                            continue
                        time += r['time']
                        identifier = r['region_identifier']
                        if backed:
                            adata_obsm.create_dataset(identifier, data=r['BindingScore'])
                            adata_obsm[identifier].attrs['encoding-type'] = 'array'
                            adata_obsm[identifier].attrs['encoding-version'] = '0.2.0'
                            # adata.obsm[identifier] = r['BindingScore']
                        else:
                            adata.obsm[identifier] = r['BindingScore']
                        # if motif_scanner is not None:
                        #     sites = r['sites']
                        #     sites['Chromosome'] = sites['Chromosome'].astype('str')
                        #     sites['Strand'] = sites['Strand'].astype('str')
                        #     sites['Score'] = sites['Score'].astype('float')
                        #
                    sys.stdout.flush()
                    gc.collect()
                    p_list.remove(p)
                    bar.update(len(rs))
                    del rs, p
                    if len(p_list) <= nCores:
                        break


        if motif_scanner is not None:
            pd.concat(index_df, axis=0).to_hdf(os.path.join(os.path.dirname(printer.file_path),
                                                            "%s_sites.h5ad" % save_key), 'site_id',
                                               mode='a', complevel=4, append=True)



        for p in as_completed(p_list):
            rs = p.result()
            for r in rs:
                if 'BindingScore' not in r:
                    print(r, "error?")
                    continue
                time += r['time']
                identifier = r['region_identifier']
                if backed:
                    adata_obsm.create_dataset(identifier, data=r['BindingScore'])
                    adata_obsm[identifier].attrs['encoding-type'] = 'array'
                    adata_obsm[identifier].attrs['encoding-version'] = '0.2.0'
                    # adata.obsm[identifier] = r['BindingScore']
                else:
                    adata.obsm[identifier] = r['BindingScore']
                # if motif_scanner is not None:
                #     sites = r['sites']
                #     sites['Chromosome'] = sites['Chromosome'].astype('str')
                #     sites['Strand'] = sites['Strand'].astype('str')
                #     sites['Score'] = sites['Score'].astype('float')
                #
            sys.stdout.flush()
            gc.collect()
            bar.update(len(rs))
            del rs
        bar.close()
        pool.shutdown(wait=True)
        print ("finishes")
        if backed:
            adata.close()
            if open_anndata:
                adata = snap.read(save_path)
        printer.bindingscoreadata[save_key] = adata
        # return adata

def _get_footprint_score_onescale(region_identifiers,
                                  cell_grouping,
                                  Tn5Bias,  # Numeric vector of predicted Tn5 bias for the current region
                                  regions,  # pyRanges object for the current region
                                  footprintRadius,
                                  flankRadius,
                                  smoothRadius,
                                  mode,
                                  id_,
                                  unique_string,
                                  return_pval,
                                  bigwigmode=False):
    result = None
    if not bigwigmode:
        insertion_profile = globals()[unique_string + "insertion_profile"]
        printer = PyPrinter(insertion_profile=insertion_profile)
    else:
        dict1 = globals()[unique_string + "group_bigwig"]

    dispModels = globals()[unique_string + "_dispModels"]

    # Create a fake pyprinter object to pass on
    # Why we need a fake one?
    # Because these functions are created to be compatible with PyPrinter centered objects
    # But it would be slow to pass the real printer project, because python would pickle everything

    for i, region_identifier in enumerate(region_identifiers):
        if i % 1000 == 0 & i > 0:
            print (i)
        # assumes the cell_grouping
        if bigwigmode:
            atac = _get_group_atac_bw(dict1, cell_grouping, regions.iloc[i])
        else:
            atac = _get_group_atac(printer, cell_grouping, regions.iloc[i])
        r, mode = getFootprint.regionFootprintScore(
            atac,
            Tn5Bias[i],
            dispModels[str(mode)],
            footprintRadius,
            flankRadius,
            mode,
            smoothRadius,
            return_pval)
        if result is None:
            result = np.zeros((len(regions), r.shape[0], r.shape[-1]))
        result[i, :, :] = r
    return result, mode, id_

def _unify_modes(modes, footprintRadius, flankRadius, smoothRadius):
    if modes is None:
        modes = np.arange(2, 101)
    # we want a list of modes, but also compatible with one mode
    if type(modes) is int:
        modes = [modes]

    # If no radius provided, reuse the modes
    if footprintRadius is None:
        footprintRadius = modes
    if flankRadius is None:
        flankRadius = modes

    if smoothRadius is None:
        smoothRadius = [int(x / 2) for x in modes]


    # Again, expct to see a list, if not, create one
    if type(footprintRadius) is int:
        footprintRadius = [footprintRadius]
    if type(flankRadius) is int:
        flankRadius = [flankRadius]
    if type(smoothRadius) is int:
        smoothRadius = [smoothRadius] * len(footprintRadius)


    return modes, footprintRadius, flankRadius, smoothRadius

def get_footprint_score(printer: PyPrinter,
                        cell_grouping: list[str] | list[list[str]] | np.ndarray | 'bigwig',
                        group_names: list[str] | str,
                        regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
                        region_width: int | None = None,
                        modes: int | list[int] | np.ndarray | None = None, # int or list of int. This is used for retrieving the correct dispersion model.
                        footprintRadius: int | list[int] | np.ndarray | None = None, # Radius of the footprint region
                        flankRadius: int | list[int] | np.ndarray | None = None, # Radius of the flanking region (not including the footprint region)
                        smoothRadius: int |  list[int] | np.ndarray | None  = 5, # Radius of the smoothing region (not including the footprint region)
                        nCores: int = 16, # Number of cores to use
                        save_key: str = None,
                        backed: bool = True,
                        overwrite: bool = False,
                        chunksize: int | None = None,
                        buffer_size: int | None = None,
                        return_pval: bool = True,
                        collapse_barcodes: bool = False,
                        ):
    """
    Get footprint score for a given region and cell group
    , stores the result in the anndata object.
    The resulting anndata object will be linked to the printer object with the key 'save_key'.
    New version updated to use footprint_generator as an underlying structure

    Parameters
    ----------
    printer: PyPrinter
        The printer object you generated by `scprinter.pp.import_fragments` or loaded by `scprinter.load_printer`
    cell_grouping: list[list[str]] | list[str] | np.ndarray
        The cell grouping you want to visualize, specifiec by a list of the cell barcodes belong to this group, e.g.
        `['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112', 'ACAGTGGT,ACAGTGGT,ATCACGTT,BUENSS112', 'ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']`.  If you want to visualize multiple groups, you can provide a list of lists, e.g.
        `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`.
    group_names: list[str] | str
        The name of the group you want to visualize.
        If you want to visualize multiple groups, you can provide a list of names, e.g. `['group1', 'group2']`
    regions: str | pd.DataFrame | pyranges.PyRanges | list[str]
        The genomic regions you want to calculate the binding score.
        You can provide a string of the path to the region bed file,
        a pandas dataframe with the first three columns correspond to [chrom, start, end], a pyranges object of the regions,
        or a list of region identifiers, e.g. `['chr1:1000-2000', 'chr1:3000-4000']`
        or a list of gene identifiers, e.g. `['Gene:CTCF', 'Gene:GATA1']`,
        the transcript start site will be used as the region center
    region_width: int | None
        The width of the region.
        The regions you provided would be resized to this width with fixed center point of each region
    modes: int | list[int] | np.ndarray
        The modes you want to calculate the footprint score for.
    footprintRadius: int | list[int] | np.ndarray | None
        The footprint radius you want to use for the footprint score calculation.
        When passing None, it will use the same value as the modes. Default is None
    flankRadius: int | list[int] | np.ndarray | None
        The flanking radius you want to use for the footprint score calculation.
        When passing None, it will use the same value as the modes. Default is None
    nCores: int
        Number of parallel process to use
    save_key: str
        The name of the anndata object you want to save the binding score result to.
    backed: bool
        Whether to save the result to a backed anndata object.
    overwrite: bool
        Whether to overwrite if there is an existing result.


    Returns
    -------
    AnnData
        This AnnData can be accessed through `printer.footprintsadata[save_key]`
        The Anndata object stores `n_obs x n_var` matrix, where `n_obs` is the number of groups, `n_vars` is the number of regions.
        The `obs` stores the group associated properties, the `var` stores the region associated properties.
        The `obsm` stores the footprint matrix, with the key of region_identifiers such as `chr1:1000-2000`.
        The value of each `obsm` has the shape of `n_obs x n_modes x region_width`.
        The `uns` stores the model parameters.

    """
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        bigwigmode = False
        modes, footprintRadius, flankRadius, smoothRadius = _unify_modes(modes, footprintRadius, flankRadius, smoothRadius)
        if cell_grouping == 'bigwig':
            bigwigmode = True
            print ("footprint from bigwig mode")

        if type(group_names) not in [np.ndarray, list]:
            group_names = [group_names]
            if cell_grouping != 'bigwig':
                cell_grouping = [cell_grouping]
        if bigwigmode:
            if any([name not in printer.insertion_file.uns['group_bigwig'] for name in group_names]):
                raise ValueError(f"group_names not in bigwig")
            cell_grouping = group_names

        regions = regionparser(regions, printer, region_width)
        region_identifiers = df2regionidentifier(regions)
        save_path = os.path.join(os.path.dirname(printer.file_path),
                                 "%s.h5ad" % save_key) if backed else 'None'
        if 'footprints' not in printer.uns:
            printer.uns['footprints'] = {}

        if save_key is None:
            save_key = "FootPrints"
        if save_key in printer.uns['footprints'] or os.path.exists(save_path):
            # Check duplicates...
            if not overwrite:
                print("detected %s, not allowing overwrite" % save_key)
                return
            else:
                try:
                    printer.footprintsadata[save_key].close()
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    del printer.footprintsadata[save_key]
                except:
                    pass

        width = np.array(regions.iloc[0])
        width = width[2] - width[1]
        estimated_filesize = width * len(regions) * len(group_names) * len(modes) * 32 * 1.16e-10
        print ("estimated file size: %.2f GB" %(estimated_filesize))

        # results will be saved in a new key
        print("Creating %s in printer.footprintsadata" % save_key)


        print("obs=groups, var=regions")
        df_obs = pd.DataFrame({'name': group_names, 'id': np.arange(len(group_names))})
        df_obs.index = group_names
        df_var = regions
        df_var['identifier'] = region_identifiers
        df_var.index = region_identifiers
        if backed:
            adata = snap.AnnData(filename=save_path, X=csr_matrix((len(group_names),
                                                                   len(regions))),
                                 obs=df_obs, var=df_var)
            adata.obs_names = [str(xx) for xx in group_names]
            adata.var_names = region_identifiers

        else:
            adata = anndata.AnnData(X=csr_matrix((len(group_names),
                                                  len(regions))),
                                    obs=df_obs, var=df_var)
        adata.uns['scales'] = np.array(modes)
        a = printer.uns['footprints']
        a[save_key] = str(save_path)
        printer.uns['footprints'] = a

        if chunksize is None:
            small_chunk_size = min(max(int(16000 / len(group_names)), 100), 10000)
        else:
            small_chunk_size = chunksize
        if buffer_size is None:
            collect_num  = int(16000 / len(group_names) * 100  / small_chunk_size)
        else:
            collect_num = buffer_size
        print (small_chunk_size, collect_num)
        threadpool = ThreadPoolExecutor(max_workers=1)

        modes = list(modes)

        if backed:
            adata.close()
            adata = h5py.File(save_path, 'r+', locking=False)
            adata_obsm = adata['obsm']
        else:
            adata_obsm = adata.obsm
            # return

        def write(multimode_footprints, region_identifiers, obsm, backed):
            if backed:
                for result, id in zip(multimode_footprints, region_identifiers):
                    obsm.create_dataset(id, data=result)
                    obsm[id].attrs['encoding-type'] = 'array'
                    obsm[id].attrs['encoding-version'] = '0.2.0'
            else:
                for result, regionID in zip(multimode_footprints, region_identifiers):
                    obsm[regionID] = result
            del multimode_footprints
            sys.stdout.flush()
            gc.collect()

        generator = footprint_generator(printer, cell_grouping,
                                                     regions, region_width,
                                                     modes, footprintRadius,
                                                     flankRadius, smoothRadius,
                                        nCores, small_chunk_size, collect_num,
                                        return_pval=return_pval, bigwigmode=bigwigmode,
                                        collapse_barcodes=collapse_barcodes)

        for multimode_footprints, regionID in generator:
            threadpool.submit(write, multimode_footprints, regionID, adata_obsm, backed)

        sys.stdout.flush()

        threadpool.shutdown(wait=True)
        if backed:
            adata.close()
            adata = snap.read(save_path)
        printer.footprintsadata[save_key] = adata
        # return adata

def footprint_generator(printer: PyPrinter,
                        cell_grouping: list[str] | list[list[str]] | np.ndarray ,
                        regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
                        region_width: int | None = None,
                        modes: int | list[int] | np.ndarray | None = None, # int or list of int. This is used for retrieving the correct dispersion model.
                        footprintRadius: int | list[int] | np.ndarray | None = None, # Radius of the footprint region
                        flankRadius: int | list[int] | np.ndarray | None = None, # Radius of the flanking region (not including the footprint region)
                        smoothRadius: int | list[int] | np.ndarray | None = 5, # Radius of the smoothing region (not including the footprint region)
                        nCores: int = 16, # Number of cores to use
                        chunk_size: int = 8000, # Number of regions to process in each chunk
                        buffer_size: int = 100, # Buffer Size for processed chunks in memory
                        async_generator: bool = True, # Whether to use asyncronous processing
                        verbose=True,
                        return_pval: bool = True,
                        bigwigmode=False,
                        collapse_barcodes=False,
                        ):
    """
    The underlying function for generating footprints. This function will return a generator that yields footprints for each region.
    The calculation of footprints will be on-going even when the calculated footprints are not fetched or downstrea procssed
    with a buffer size (collect_num) to avoid memory overflow.

    Parameters
    ----------
    printer: PyPrinter
        The printer object you generated by `scprinter.pp.import_fragments` or loaded by `scprinter.load_printer`
    cell_grouping: list[list[str]] | list[str] | np.ndarray
        The cell grouping you want to visualize, specifiec by a list of the cell barcodes belong to this group, e.g.
        `['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112', 'ACAGTGGT,ACAGTGGT,ATCACGTT,BUENSS112', 'ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']`.  If you want to visualize multiple groups, you can provide a list of lists, e.g.
        `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`.
    regions: str | pd.DataFrame | pyranges.PyRanges | list[str]
        The genomic regions you want to calculate the binding score.
        You can provide a string of the path to the region bed file,
        a pandas dataframe with the first three columns correspond to [chrom, start, end], a pyranges object of the regions,
        or a list of region identifiers, e.g. `['chr1:1000-2000', 'chr1:3000-4000']`
        or a list of gene identifiers, e.g. `['Gene:CTCF', 'Gene:GATA1']`,
        the transcript start site will be used as the region center
    region_width: int | None
        The width of the region.
        The regions you provided would be resized to this width with fixed center point of each region
    modes: int | list[int] | np.ndarray
        The modes you want to calculate the footprint score for.
    footprintRadius: int | list[int] | np.ndarray | None
        The footprint radius you want to use for the footprint score calculation.
        When passing None, it will use the same value as the modes. Default is None
    flankRadius: int | list[int] | np.ndarray | None
        The flanking radius you want to use for the footprint score calculation.
        When passing None, it will use the same value as the modes. Default is None
    nCores: int
        Number of parallel process to use
    chunk_size: int
        Number of regions to process in each chunk
    buffer_size: int
        Buffer Size for the results of scales x chunks in memory. (So if buffer_size=99,
        we store one chunk of 99 scales)
    async_generator: bool
        Whether to use asyncronous processing, when True,
        the returned footprints will not keep the same order as the peaks_iter. when False,
        the returned footprints will keep the same order as the peaks_iter
    Returns
    -------
    generator
        A generator that yields footprints for each chunked regions (order of the chunks can be shuffled depending on
        the param `async_generator`).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        unique_string = printer.unique_string




        # global insertion_profile, dispModels

        modes, footprintRadius, flankRadius, smoothRadius = _unify_modes(modes, footprintRadius, flankRadius, smoothRadius)


        if type(cell_grouping) not in [np.ndarray, list]:
            cell_grouping = [cell_grouping]

        regions = regionparser(regions, printer, region_width)
        region_identifiers = df2regionidentifier(regions)

        assert printer.dispersionModel is not None, "dispersion Model not loaded, run printer.load_disp_model first"

        dispModels = printer.dispersionModel
        insertion_profile = printer.fetch_insertion_profile()



        if not bigwigmode:
            cell_grouping = cell_grouping2cell_grouping_idx(printer,
                                                            cell_grouping)

        if collapse_barcodes:
            new_insertion_profile = {}
            new_cell_grouping = [[i] for i in range(len(cell_grouping))]
            for key in tqdm(insertion_profile, desc='collapsing_barcode'):
                xx = [csc_matrix(insertion_profile[key][barcodes].sum(axis=0)) for barcodes in cell_grouping]
                new_insertion_profile[key] = vstack(xx)
            insertion_profile = new_insertion_profile
            cell_grouping = new_cell_grouping
        if not bigwigmode:
            globals()[unique_string + "insertion_profile"] = insertion_profile
        else:
            globals()[unique_string + "group_bigwig"] = printer.uns['group_bigwig']

        globals()[unique_string + "_dispModels"] = dispModels

        Tn5Bias = getBias.getPrecomputedBias(printer.insertion_file.uns['bias_path'],
                                             regions,
                                             savePath=None)

        pool = ProcessPoolExecutor(max_workers=nCores)
        modes = list(modes)
        multimode_footprints_all = {}
        p_list = []


        Task_generator = itertools.product(range(0, len(regions), chunk_size), # region iterator,
                                           zip(modes, footprintRadius, flankRadius, smoothRadius), # footprint scale iterator)
                                                )

        return_order = list(range(0, len(regions), chunk_size))
        # print (return_order)
        next_return = return_order.pop(0)

        bar1 = trange(len(regions) * len(modes), desc='Submitting jobs', disable=not verbose)
        bar2 = trange(len(regions) * len(modes), desc='collecting multi-scale footprints', disable=not verbose)

        submitted_job_num = 0
        for i, (mode, r1, r2, r3) in Task_generator:
            slice_ = slice(i, i + chunk_size)
            if i not in multimode_footprints_all:
                multimode_footprints_all[i] = [None] * len(modes)
            p_list.append(pool.submit(
                _get_footprint_score_onescale,
                region_identifiers[slice_],
                cell_grouping,
                Tn5Bias[slice_],
                regions.iloc[slice_],
                r1,
                r2,
                r3,
                mode,
                i,
                unique_string,
                return_pval,
                bigwigmode
            ))
            if verbose:
                bar1.update(len(Tn5Bias[slice_]))
            submitted_job_num += len(Tn5Bias[slice_])
            final_collect = (submitted_job_num == len(regions) * len(modes))

            # When there are more than buffer_size jobs in the queue, or we are at the final collect (all jobs submitted)
            if (len(p_list) > buffer_size) or final_collect:
                # print ("start collecting")
                for p in as_completed(p_list):
                    results, mode, id = p.result()
                    p_list.remove(p)
                    multimode_footprints_all[id][modes.index(mode)] = results
                    if verbose:
                        bar2.update(len(results))

                    # Which chunk of region to return, if it's async, it will say whoever is ready to return we return
                    # if it's not async, we return the next chunk of region decided by next_return
                    id_for_return = id if async_generator else next_return
                    # When we fetched all scales for a region, yield the stacked results
                    if not any([xx is None for xx in multimode_footprints_all[id_for_return]]):
                        multimode_footprints = np.stack(multimode_footprints_all[id_for_return], axis=2)
                        slice_ = slice(id_for_return, id_for_return + chunk_size)
                        yield multimode_footprints, region_identifiers[slice_]
                        del multimode_footprints_all[id_for_return]
                        if not async_generator:
                            try:
                                next_return = return_order.pop(0)
                            except:
                                next_return = -1

                    # When half of the buffer_size jobs are done and we still have jobs to submit,
                    # break the loop and submit more jobs
                    if len(p_list) <= (buffer_size / 2) and not final_collect:
                        break
        if verbose:
            bar1.close()
            bar2.close()

def _stats_inter_versus_intra(X: np.ndarray,
                              Y: np.ndarray,
                              metric:str='euclidean',
                              summary = np.mean,
                              sample_axis=0,
                              ):
    """
    
    Parameters
    ----------
    X - has shape (n, scale, width)
    Y - has shape (m, scale, width)

    Returns
    -------
    dis (X to Y) / [dis (X to X) + dis (Y to Y) ]
    
    """
    x1 = np.mean(X, axis=0)
    y1 = np.mean(Y, axis=0)
    mask = np.isnan(x1) | np.isnan(y1) | np.isinf(x1) | np.isinf(y1)
    X[:, mask] = 0
    Y[:, mask] = 0

    if sample_axis == 0:
        X = X.reshape(X.shape[0], -1)
        Y = Y.reshape(Y.shape[0], -1)
    else:
        X = X.reshape(-1, X.shape[-1])
        Y = Y.reshape(-1, Y.shape[-1])
    inter = pairwise_distances(X, Y, metric=metric)

    intra1 = pairwise_distances(X, metric=metric)
    intra2 = pairwise_distances(Y, metric=metric)
    # print (inter, intra1, intra2)
    # print (inter.shape, intra1.shape, intra2.shape)
    return summary(inter) / (summary(intra1)+summary(intra2)+1) #/ (summary(np.linalg.norm(intra1, axis=1)) + summary(np.linalg.norm(intra2, axis=1)) + 1)


def get_insertions(printer: PyPrinter,
                    cell_grouping: list[str] | list[list[str]] | np.ndarray,
                    group_names: list[str] | str,
                    regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
                    region_width: int | None = None,
                    save_key: str = None,
                    backed: bool = True,
                    overwrite: bool = False,
                    summarize_func=None):
    """
    Get the insertion profile for given regions and cell group
    Parameters
    ----------
    printer
    cell_grouping
    group_names
    regions
    region_width
    save_key
    backed
    overwrite
    summarize_func

    Returns
    -------

    """

    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")

        if type(group_names) not in [np.ndarray, list]:
            group_names = [group_names]
            cell_grouping = [cell_grouping]

        regions = regionparser(regions, printer, region_width)
        region_identifiers = df2regionidentifier(regions)
        save_path = os.path.join(os.path.dirname(printer.file_path),
                                 "%s.h5ad" % save_key) if backed else 'None'
        if 'insertions' not in printer.uns:
            printer.uns['insertions'] = {}

        if save_key is None:
            save_key = "Insertions"
        if save_key in printer.uns['insertions'] or os.path.exists(save_path):
            # Check duplicates...
            if not overwrite:
                print("detected %s, not allowing overwrite" % save_key)
                return
            else:
                try:
                    printer.insertionadata[save_key].close()
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    del printer.insertionadata[save_key]
                except:
                    pass

        width = np.array(regions.iloc[0])
        width = width[2] - width[1]

        # results will be saved in a new key
        print("Creating %s in printer.insertionadata" % save_key)


        print("obs=groups, var=regions")
        df_obs = pd.DataFrame({'name': group_names, 'id': np.arange(len(group_names))})
        df_obs.index = group_names
        df_var = regions
        df_var['identifier'] = region_identifiers
        df_var.index = region_identifiers
        if backed:
            adata = snap.AnnData(filename=save_path, X=csr_matrix((len(group_names),
                                                                   len(regions))),
                                 obs=df_obs, var=df_var)
            adata.obs_names = [str(xx) for xx in group_names]
            adata.var_names = region_identifiers

        else:
            adata = anndata.AnnData(X=csr_matrix((len(group_names),
                                                  len(regions))),
                                    obs=df_obs, var=df_var)

        a = printer.uns['footprints']
        a[save_key] = str(save_path)
        printer.uns['footprints'] = a
        cell_grouping = cell_grouping2cell_grouping_idx(printer,
                                                        cell_grouping)
        if backed:
            adata.close()
            adata = h5py.File(save_path, 'r+', locking=False)
            adata_obsm = adata['obsm']
        else:
            adata_obsm = adata.obsm

        for i, region_identifier in enumerate(tqdm(region_identifiers)):
            atac = _get_group_atac(printer, cell_grouping, regions.iloc[i])
        #     if summarize_func is not None:
        #         atac = summarize_func(atac)
        #     adata_obsm[region_identifier] = atac
        # printer.insertionadata[save_key] = adata