import time

import pandas as pd

pd.set_option("chained_assignment", None)
import gzip
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pybedtools
from tqdm.auto import trange

from .utils import *


# Input: pd.Dataframe with column 'barcode' and 'group'
# Output: a dictionary that maps barcode to its group_id (int),
# and a np.array of unique group in the order corresponding to group_id
def constructGroupBarcodesDict(barcodeGroups, existing_groups):
    unique_group = np.unique(barcodeGroups["group"])
    # for new groups, append it at the last, such that previous results remain the same
    if len(existing_groups) > 1:
        new_groups = unique_group[~np.isin(unique_group, existing_groups)]
        new_unique_group = np.concatenate([existing_groups, new_groups])
    else:
        new_unique_group = unique_group

    group2id = {g: i for i, g in enumerate(new_unique_group)}

    # only build dictionary for the barcodeGroups provided
    dict1 = {}
    for b, g in zip(barcodeGroups["barcode"], barcodeGroups["group"]):
        if b not in dict1:
            dict1[b] = []
        dict1[b].append(group2id[g])

    return dict1


def fragmentsToInsertion(
    frags,  # Chunks of fragments
    maxFragLength,  # maximum fragment length
    regions,  # pybedtools object
    region_columns,  # column names of the region (used for bedtools later)
    extra_plus_shift,  # extra shift for + strand
    extra_minus_shift,  # extra shift for - strand
    barcodeGroups_dict,  # barcodegroups_dictionary genertted from df
    chunkSize,  # chunkSize for region
):
    """
    Convert fragments to insertion

    Parameters
    ----------
    frags
        Path to the fragments file
    maxFragLength
        Maximum fragment length
    regions
        Path to the regions file
    region_columns
        Column names of the region (used for bedtools later)
    extra_plus_shift
        Extra shift for + strand
    extra_minus_shift
        Extra shift for - strand
    barcodeGroups_dict
        Barcodegroups_dictionary generated from df
    chunkSize
        Chunk size for region

    Returns
    -------

    """
    regions = pybedtools.BedTool.from_dataframe(regions)
    # only keep barcodes within the provided group list
    Filter = np.array([xx in barcodeGroups_dict for xx in frags[3]])
    if maxFragLength is not None:
        maxLengthFilter = (frags[2] - frags[1]) <= maxFragLength
        Filter = np.logical_and(Filter, maxLengthFilter)
    frags = frags[Filter]

    # do the extra shift
    frags.loc[:, 1] = frags.loc[:, 1] + extra_plus_shift
    frags.loc[:, 2] = frags.loc[:, 2] + extra_minus_shift

    # to insertion
    insertion_left = frags[[0, 1, 3]]
    insertion_right = frags[[0, 2, 3]]
    insertion_left.columns = insertion_right.columns = [
        "Chromosome",
        "Start",
        "Barcode",
    ]
    insertion = pd.concat([insertion_left, insertion_right], axis=0)

    insertion["End"] = insertion["Start"] + 1
    insertion = insertion[["Chromosome", "Start", "End", "Barcode"]]

    # pybedtools intersection
    insertion = pybedtools.BedTool.from_dataframe(insertion)
    insertion = insertion.intersect(regions, wa=True, wb=True).to_dataframe(
        names=["Insertion_Chrom", "Insertion_Start", "Insertion_End", "Barcode"] + region_columns
    )
    # If no overlap, return an empty dict
    if len(insertion) == 0:
        return {}

    # stores group info
    insertion["Group"] = [barcodeGroups_dict[xx] for xx in insertion["Barcode"]]
    # calculate the relative position
    insertion["position"] = insertion["Insertion_Start"] - insertion["Start"]
    # Throw away the 'Barcode' Column to save space for explode operation later
    insertion = insertion[["region_id", "position", "Group"]]
    # for barcode with multiple groups, explode will make copies of them and spread to different group
    insertion = insertion.explode("Group")

    # merge records with the same region_id, position, group
    # Will need to do it again after finish all fragchunks (unmerged records across different fragchunks)
    insertion["count"] = 1
    insertion = insertion[["region_id", "position", "Group", "count"]]
    summary = (
        insertion.groupby(by=["position", "Group", "region_id"])
        .sum()
        .reset_index()
        .sort_values(by=["Group", "position"])
    )
    summary = summary[["region_id", "position", "Group", "count"]]

    # indexed by chunk ID,
    summary.index = np.floor(np.copy(summary["region_id"]) / chunkSize).astype("int")
    uniq_chunk = np.unique(summary.index)

    # separate results by chunk ID
    result = {chunkInd: summary.loc[chunkInd] for chunkInd in uniq_chunk}
    pybedtools.helpers.cleanup()
    return result


# Merge count with the same reigon_id & position_id & group_id
def summarizeInsertion(insertion):
    summary = (
        insertion.groupby(by=["position", "Group", "region_id"])
        .sum()
        .reset_index()
        .sort_values(by=["Group", "position"])
    )
    summary = summary[["region_id", "position", "Group", "count"]]
    summary.index = np.copy(summary["region_id"])
    dict1 = {}
    for regionInd in np.unique(summary["region_id"]):
        a = summary.loc[regionInd][["position", "Group", "count"]]
        a = np.array(a)
        dict1[regionInd] = a
    return dict1


def mergeRegionCountTensor(count1, count2):
    # each count is three col of pos, group, count, can be treated as coo.
    df = pd.DataFrame(
        np.concatenate([count1, count2], axis=0), columns=["position", "Group", "count"]
    )
    summary = (
        df.groupby(by=["position", "Group"])
        .sum()
        .reset_index()
        .sort_values(by=["Group", "position"])
    )
    summary = summary[["position", "Group", "count"]]
    return np.array(summary)


# count how many lines in a file
def countLine(filename):
    if ".gz" in filename:
        line_count = sum(1 for i in gzip.open(filename, "rb"))
    else:
        line_count = sum(1 for i in open(filename, "rb"))
    return line_count


def computeCountTensor(
    pathToFrags,  # Path or list of paths to fragments file
    regions,  # path to a bed file,
    # pandas dataframe,
    # or string like "chr1:1-10000"
    # or list of string ["chr1:1-10000", "chr12:1-10000"]  specifying the reigons
    barcodeGroups,  # dataframe or list of dataframe specifying membership of barcodes in pseudobulks.
    saveDir,  # Directory to store results
    # First column is barcodes and second is groupID
    saveName="chunkedCountTensor.h5",
    maxFragLength=None,  # Fragment length upper limit
    nrows=np.Inf,  # Max number of rows when reading from fragments file
    chunkSize=2000,  # Chunk size for parallel processing of regions (I want to remove this arg)
    fragchunkSize=1000000,
    nCores=16,  # Number of cores to use
    returnCombined=False,
    # Whether to return the combined result for all chunks. Set it to False when data is too big,
    plus_shift=4,
    minus_shift=-5,
    append_mode=False,  # When True, it means, the new results will be appended (either more reads (mode 3)
    # / more groups (mode 2) / more regions (mode 1))
):
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    tmp_hdf5_path = os.path.join(saveDir, "chunks_temp.h5")
    final_hdf5_path = os.path.join(saveDir, "%s" % saveName)

    # remove tmp hdf5
    if os.path.exists(tmp_hdf5_path):
        os.remove(tmp_hdf5_path)
    if os.path.exists(final_hdf5_path):
        os.remove(final_hdf5_path)

    # hoping for a list, but if it's a string, treated it as a list
    if type(pathToFrags) is str:
        pathsToFrags = [pathToFrags]
    else:
        pathsToFrags = pathToFrags

    # hoping for a pandas table frame, if a list, concat it
    if type(barcodeGroups) is list:
        barcodeGroups = pd.concat(barcodeGroups, axis=0)

    barcodeGroups.columns = ["barcode", "group"]
    # get the regions bed file
    regions = regionparser(regions)

    # This region_id wouldn't be stored in the final results or used anywhere else.
    # It's just for internal chunk split purpose
    regions["region_id"] = np.arange(len(regions))

    # remove duplicated files to avoid crashing
    final_hdf5 = h5py.File(final_hdf5_path, "a" if append_mode else "w")

    # region_identifier is the string like chr1:0-1000 that identifies each region.
    region_identifier = df2regionidentifier(regions)

    uniq_group, existing_groups, new_group, uniq_region, existing_region, new_region = (
        Unify_meta_info(
            final_hdf5,
            addition_feats=[np.unique(barcodeGroups["group"]), region_identifier],
            entries=["group", "region"],
            dtypes=["str", "str"],
        )
    )

    if "regions" in final_hdf5.keys():
        # delete existing groups and regions, since we'll rewrite later.
        del final_hdf5["regions"], final_hdf5["group"]

    if append_mode == 1:
        # append region mode, check if there are new groups:
        assert len(new_group) == 0, (
            "append_region mode, but observed changed cell grouping: %s ..." % new_group[0]
        )
        assert len(new_region) == len(uniq_region), (
            "append mode, cannot have overlap regions: %s"
            % uniq_region[np.isin(uniq_region, existing_region)][0]
        )
    elif append_mode == 2:
        # append group mode, check if there are new regions
        assert len(new_region) == 0, (
            "append_group mode, but observed changed regions: %s..." % new_region[0]
        )
        assert len(new_group) == len(uniq_group), (
            "append mode, cannot have overlap groups: %s"
            % uniq_group[np.isin(uniq_group, existing_groups)][0]
        )
    elif append_mode == 3:
        assert len(new_group) == 0 and len(new_region) == 0, (
            "append_frags mode, but region / "
            "cell grouping changed \n "
            "new regions: %s... \n "
            "new groups: %s... \n" % (new_region[0], new_group[0])
        )

    # hdf5 in pythonwants the dtype to be specific
    groups_all = np.array(list(existing_groups) + list(uniq_group))
    groups_all = [str(xx) for xx in groups_all]
    final_hdf5.create_dataset("group", data=groups_all, dtype=h5py.special_dtype(vlen=str))
    regions_all = list(existing_region) + list(region_identifier)
    regions_all = sort_region_identifier(regions_all)
    final_hdf5.create_dataset("regions", data=regions_all, dtype=h5py.special_dtype(vlen=str))
    final_hdf5.attrs["description"] = "Stored value represents [position, group_id, count]"

    # store the additional columns names for future bedtools intersect reference_ref
    region_columns = list(regions.columns)

    barcodeGroups_dict = constructGroupBarcodesDict(barcodeGroups, existing_groups)

    # Every chunks share one hdf5 file.
    # The final results are stored in ['chunk_id']
    hdf5_storage = pd.HDFStore(tmp_hdf5_path)

    # this is equivalent to do +4/-4 shift,
    # because python is 0-based, and I'll just use the right end as index
    # meaning that I'll do [end, end+1) as insertion position, not [end-1, end)
    extra_plus_shift = 4 - plus_shift
    extra_minus_shift = -5 - minus_shift
    print("extra_shift: p/m", extra_plus_shift, extra_minus_shift)
    print("Removing frags with length >", maxFragLength, " bp")

    # Start a multiprocessing pool
    pool = ProcessPoolExecutor(max_workers=nCores)
    fragchunkSize = min(fragchunkSize, nrows)

    # for each fragments file
    for pathToFrags in pathsToFrags:
        print("Processing file:", pathToFrags.split("/")[-1])
        p_list = []

        if ".gz" in pathToFrags:
            csv_file = gzip.open(pathToFrags, "rb")
        else:
            csv_file = open(pathToFrags, "r")

        reader = pd.read_csv(csv_file, chunksize=fragchunkSize, sep="\t", header=None)
        start = time.time()
        # chrom, start, end, fragments
        # read and process chunk by chunk
        submit_total = 0
        finish_total = 0
        for chunk_fragments_count, chunk_fragments in enumerate(reader):

            if chunk_fragments_count * fragchunkSize >= nrows:
                print("reaching nrows: ", nrows, "quitting")
                break

            # chunk of fragments to insertion. fork/spawn a child process for that
            p_list.append(
                pool.submit(
                    fragmentsToInsertion,
                    chunk_fragments,
                    maxFragLength,
                    regions,
                    region_columns,
                    extra_plus_shift,
                    extra_minus_shift,
                    barcodeGroups_dict,
                    chunkSize,
                )
            )
            submit_total += 1
            print(
                "submitting %d jobs, takes: %.2f s\r"
                % (chunk_fragments_count, time.time() - start),
                end="",
            )

            # We got a lot in the list, let's finish some first
            if len(p_list) > nCores * 2:
                while len(p_list) > nCores:
                    for p in as_completed(p_list):
                        finish_total += 1
                        insertions = p.result()
                        for chunk in insertions:
                            xx = insertions[chunk]
                            if type(xx) is pd.core.series.Series:
                                xx = pd.DataFrame([xx.rename(None)])
                            hdf5_storage.put(
                                "chunk_%d" % (chunk),
                                xx,
                                format="table",
                                append=True,
                                index=False,
                            )
                        p_list.remove(p)
                        del p

        print()
        csv_file.close()

        bar = trange(submit_total, desc=" - Processing ", leave=True)
        bar.update(finish_total)
        # whatever fragchunks finished first, we'll process it and store it
        for p in as_completed(p_list):
            bar.update(1)
            insertions = p.result()
            for chunk in insertions:
                hdf5_storage.put(
                    "chunk_%d" % (chunk),
                    insertions[chunk],
                    format="table",
                    append=True,
                    index=False,
                )

    # once everything finished:
    # read back the concatenated table, and reorganize:
    regions = regions
    chunks = int(math.ceil(len(regions) / chunkSize))
    print(chunks)
    p_list = []
    bar = trange(chunks)
    for i in range(chunks):
        try:
            insertion = hdf5_storage.get("chunk_%d" % i).reset_index()
        except:
            print("No records for chunk_", i)
            continue
        # submit child process to summarize insertion profile
        p_list.append(pool.submit(summarizeInsertion, insertion))

    for p in as_completed(p_list):
        result = p.result()
        bar.update(1)
        for regionInd in result:
            # Main different is that,
            # To save space, I'm saving it as (position, group_id, count) 3-col int array
            xx = result[regionInd]
            if append_mode:
                # In append mode, if there's a region alreay
                if region_identifier[regionInd] in final_hdf5:
                    old_result = np.array(final_hdf5[region_identifier[regionInd]])
                    del final_hdf5[region_identifier[regionInd]]
                    if append_mode == 3:
                        # more reads/ fragments mode:
                        # need to add results
                        xx = mergeRegionCountTensor(old_result, result[regionInd])
                    else:
                        # more groups? just concat
                        xx = np.concatenate([old_result, result[regionInd]])
                # else: more regions? just create!
            final_hdf5.create_dataset(region_identifier[regionInd], data=xx, compression="gzip")

    pool.shutdown(wait=True)
    bar.close()
    final_hdf5.close()
    os.remove(tmp_hdf5_path)
    pybedtools.helpers.cleanup(verbose=False, remove_all=True)
    if returnCombined:
        with h5py.File(final_hdf5_path, "r") as f:
            return_list = [
                np.array(f[region_identifier[i]]) if region_identifier[i] in f else None
                for i in range(len(regions))
            ]
            return return_list


def appendCountTensor_regions(
    *args,  # dataframe or list of dataframe specifying membership of barcodes in pseudobulks.
    **kwargs,
):
    computeCountTensor(*args, append_mode=1, **kwargs)


def appendCountTensor_groups(
    *args,  # dataframe or list of dataframe specifying membership of barcodes in pseudobulks.
    **kwargs,
):
    computeCountTensor(*args, append_mode=2, **kwargs)


def appendCountTensor_frags(*args, **kwargs):
    computeCountTensor(*args, append_mode=3, **kwargs)
