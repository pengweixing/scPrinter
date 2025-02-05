#!/usr/bin/env python
# Adapted from tf-modisco command-line tool: Credit goes to Author: Jacob Schreiber <jmschreiber91@gmail.com>, Ivy Raine <ivy.ember.raine@gmail.com>
# Adding more options regarding the leiden solver

import argparse
from typing import List, Literal, Union

import h5py
import igraph as ig
import leidenalg
import modiscolite
from modiscolite.util import calculate_window_offsets


def LeidenCluster(
    affinity_mat, n_seeds=2, n_leiden_iterations=-1, resolution=1.0, solver="modularity"
):
    n_vertices = affinity_mat.shape[0]
    n_cols = affinity_mat.indptr
    sources = np.concatenate(
        [np.ones(n_cols[i + 1] - n_cols[i], dtype="int32") * i for i in range(n_vertices)]
    )

    g = ig.Graph(directed=None)
    g.add_vertices(n_vertices)
    g.add_edges(zip(sources, affinity_mat.indices))

    best_clustering = None
    best_quality = None

    for seed in range(1, n_seeds + 1):
        partition = leidenalg.find_partition(
            graph=g,
            partition_type=(
                leidenalg.RBConfigurationVertexPartition
                if solver == "rb"
                else leidenalg.ModularityVertexPartition
            ),
            weights=affinity_mat.data,
            n_iterations=n_leiden_iterations,
            initial_membership=None,
            seed=seed * 100,
            **({"resolution_parameter": resolution} if solver == "rb" else {}),
        )

        quality = np.array(partition.quality())
        membership = np.array(partition.membership)

        if best_quality is None or quality > best_quality:
            best_quality = quality
            best_clustering = membership

    return best_clustering


# tfmodisco.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar

from collections import OrderedDict, defaultdict

import numpy as np
import scipy
import scipy.sparse
from modiscolite import affinitymat, aggregator, cluster, core, extract_seqlets, util


def _density_adaptation(affmat_nn, seqlet_neighbors, tsne_perplexity):
    eps = 0.0000001

    rows, cols, data = [], [], []
    for row in range(len(affmat_nn)):
        for col, datum in zip(seqlet_neighbors[row], affmat_nn[row]):
            rows.append(row)
            cols.append(col)
            data.append(datum)

    affmat_nn = scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(affmat_nn), len(affmat_nn)), dtype="float64"
    )

    affmat_nn.data = np.maximum(np.log((1.0 / (0.5 * np.maximum(affmat_nn.data, eps))) - 1), 0)
    affmat_nn.eliminate_zeros()

    counts_nn = scipy.sparse.csr_matrix(
        (np.ones_like(affmat_nn.data), affmat_nn.indices, affmat_nn.indptr),
        shape=affmat_nn.shape,
        dtype="float64",
    )

    affmat_nn += affmat_nn.T
    counts_nn += counts_nn.T
    affmat_nn.data /= counts_nn.data
    del counts_nn

    betas = [
        util.binary_search_perplexity(tsne_perplexity, affmat_nn[i].data)
        for i in range(affmat_nn.shape[0])
    ]
    normfactors = np.array(
        [np.exp(-np.array(affmat_nn[i].data) / beta).sum() + 1 for i, beta in enumerate(betas)]
    )

    for i in range(affmat_nn.shape[0]):
        for j_idx in range(affmat_nn.indptr[i], affmat_nn.indptr[i + 1]):
            j = affmat_nn.indices[j_idx]
            distance = affmat_nn.data[j_idx]

            rbf_i = np.exp(-distance / betas[i]) / normfactors[i]
            rbf_j = np.exp(-distance / betas[j]) / normfactors[j]

            affmat_nn.data[j_idx] = np.sqrt(rbf_i * rbf_j)

    affmat_diags = scipy.sparse.diags(1.0 / normfactors)
    affmat_nn += affmat_diags
    return affmat_nn


def _filter_patterns(
    patterns, min_seqlet_support, window_size, min_ic_in_window, background, ppm_pseudocount
):
    passing_patterns = []
    for pattern in patterns:
        if len(pattern.seqlets) < min_seqlet_support:
            continue

        ppm = pattern.sequence
        per_position_ic = util.compute_per_position_ic(
            ppm=ppm, background=background, pseudocount=ppm_pseudocount
        )

        if len(per_position_ic) < window_size:
            if np.sum(per_position_ic) < min_ic_in_window:
                continue
        else:
            # do the sliding window sum rearrangement
            windowed_ic = np.sum(
                util.rolling_window(a=per_position_ic, window=window_size), axis=-1
            )

            if np.max(windowed_ic) < min_ic_in_window:
                continue

        passing_patterns.append(pattern)

    return passing_patterns


def _patterns_from_clusters(
    seqlets,
    track_set,
    min_overlap,
    min_frac,
    min_num,
    flank_to_add,
    window_size,
    bg_freq,
    cluster_indices,
    track_sign,
):
    seqlet_sort_metric = lambda x: -np.sum(np.abs(x.contrib_scores))
    num_clusters = max(cluster_indices + 1)
    cluster_to_seqlets = defaultdict(list)

    for seqlet, idx in zip(seqlets, cluster_indices):
        cluster_to_seqlets[idx].append(seqlet)

    patterns = []
    for i in range(num_clusters):
        sorted_seqlets = sorted(cluster_to_seqlets[i], key=seqlet_sort_metric)
        pattern = core.SeqletSet([sorted_seqlets[0]])

        if len(sorted_seqlets) > 1:
            pattern = aggregator.merge_in_seqlets_filledges(
                parent_pattern=pattern,
                seqlets_to_merge=sorted_seqlets[1:],
                track_set=track_set,
                metric=affinitymat.jaccard,
                min_overlap=min_overlap,
            )

        pattern = aggregator.polish_pattern(
            pattern,
            min_frac=min_frac,
            min_num=min_num,
            track_set=track_set,
            flank=flank_to_add,
            window_size=window_size,
            bg_freq=bg_freq,
        )

        if pattern is not None:
            if np.sign(np.sum(pattern.contrib_scores)) == track_sign:
                patterns.append(pattern)

    return patterns


def _filter_by_correlation(
    seqlets, seqlet_neighbors, coarse_affmat_nn, fine_affmat_nn, correlation_threshold
):
    correlations = []
    for fine_affmat_row, coarse_affmat_row in zip(fine_affmat_nn, coarse_affmat_nn):
        to_compare_mask = np.abs(fine_affmat_row) > 0
        corr = scipy.stats.spearmanr(
            fine_affmat_row[to_compare_mask], coarse_affmat_row[to_compare_mask]
        )
        correlations.append(corr.correlation)

    correlations = np.array(correlations)
    filtered_rows_mask = np.array(correlations) > correlation_threshold

    filtered_seqlets = [seqlet for seqlet, mask in zip(seqlets, filtered_rows_mask) if mask == True]

    # figure out a mapping from pre-filtering to the
    # post-filtering indices
    new_idx_mapping = np.cumsum(filtered_rows_mask) - 1
    retained_indices = set(np.where(filtered_rows_mask == True)[0])

    filtered_neighbors = []
    filtered_affmat_nn = []
    for old_row_idx, (old_neighbors, affmat_row) in enumerate(
        zip(seqlet_neighbors, fine_affmat_nn)
    ):
        if old_row_idx in retained_indices:
            filtered_old_neighbors = [
                neighbor for neighbor in old_neighbors if neighbor in retained_indices
            ]
            filtered_affmat_row = [
                affmatval
                for affmatval, neighbor in zip(affmat_row, old_neighbors)
                if neighbor in retained_indices
            ]
            filtered_neighbors_row = [
                new_idx_mapping[neighbor] for neighbor in filtered_old_neighbors
            ]
            filtered_neighbors.append(filtered_neighbors_row)
            filtered_affmat_nn.append(filtered_affmat_row)

    return filtered_seqlets, filtered_neighbors, filtered_affmat_nn


def seqlets_to_patterns(
    seqlets,
    track_set,
    track_signs=None,
    min_overlap_while_sliding=0.7,
    nearest_neighbors_to_compute=500,
    affmat_correlation_threshold=0.15,
    tsne_perplexity=10.0,
    n_leiden_iterations=-1,
    n_leiden_runs=50,
    frac_support_to_trim_to=0.2,
    min_num_to_trim_to=30,
    trim_to_window_size=20,
    initial_flank_to_add=5,
    prob_and_pertrack_sim_merge_thresholds=[(0.8, 0.8), (0.5, 0.85), (0.2, 0.9)],
    prob_and_pertrack_sim_dealbreaker_thresholds=[(0.4, 0.75), (0.2, 0.8), (0.1, 0.85), (0.0, 0.9)],
    subcluster_perplexity=50,
    merging_max_seqlets_subsample=300,
    final_min_cluster_size=20,
    min_ic_in_window=0.6,
    min_ic_windowsize=6,
    ppm_pseudocount=0.001,
    leiden_resolution=1.0,
    leiden_solver="modularity",
):
    bg_freq = np.mean([seqlet.sequence for seqlet in seqlets], axis=(0, 1))

    seqlets_sorter = lambda arr: sorted(arr, key=lambda x: -np.sum(np.abs(x.contrib_scores)))

    seqlets = seqlets_sorter(seqlets)

    for round_idx in range(2):
        if len(seqlets) == 0:
            return None

        # Step 1: Generate coarse resolution
        coarse_affmat_nn, seqlet_neighbors = affinitymat.cosine_similarity_from_seqlets(
            seqlets=seqlets, n_neighbors=nearest_neighbors_to_compute, sign=track_signs
        )

        # Step 2: Generate fine representation
        fine_affmat_nn = affinitymat.jaccard_from_seqlets(
            seqlets=seqlets,
            seqlet_neighbors=seqlet_neighbors,
            min_overlap=min_overlap_while_sliding,
        )

        if round_idx == 0:
            filtered_seqlets, seqlet_neighbors, filtered_affmat_nn = _filter_by_correlation(
                seqlets,
                seqlet_neighbors,
                coarse_affmat_nn,
                fine_affmat_nn,
                affmat_correlation_threshold,
            )
        else:
            filtered_seqlets = seqlets
            filtered_affmat_nn = fine_affmat_nn

        del coarse_affmat_nn
        del fine_affmat_nn
        del seqlets

        # Step 4: Density adaptation
        csr_density_adapted_affmat = _density_adaptation(
            filtered_affmat_nn, seqlet_neighbors, tsne_perplexity
        )

        del filtered_affmat_nn
        del seqlet_neighbors

        # Step 5: Clustering
        cluster_indices = LeidenCluster(
            csr_density_adapted_affmat,
            n_seeds=n_leiden_runs,
            n_leiden_iterations=n_leiden_iterations,
            resolution=leiden_resolution,
            solver=leiden_solver,
        )

        del csr_density_adapted_affmat

        patterns = _patterns_from_clusters(
            filtered_seqlets,
            track_set=track_set,
            min_overlap=min_overlap_while_sliding,
            min_frac=frac_support_to_trim_to,
            min_num=min_num_to_trim_to,
            flank_to_add=initial_flank_to_add,
            window_size=trim_to_window_size,
            bg_freq=bg_freq,
            cluster_indices=cluster_indices,
            track_sign=track_signs,
        )

        # obtain unique seqlets from adjusted motifs
        seqlets = list(dict([(y.string, y) for x in patterns for y in x.seqlets]).values())

    del seqlets

    merged_patterns, pattern_merge_hierarchy = aggregator._detect_spurious_merging(
        patterns=patterns,
        track_set=track_set,
        perplexity=subcluster_perplexity,
        min_in_subcluster=max(final_min_cluster_size, subcluster_perplexity),
        min_overlap=min_overlap_while_sliding,
        prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
        prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
        min_frac=frac_support_to_trim_to,
        min_num=min_num_to_trim_to,
        flank_to_add=initial_flank_to_add,
        window_size=trim_to_window_size,
        bg_freq=bg_freq,
        max_seqlets_subsample=merging_max_seqlets_subsample,
        n_seeds=n_leiden_runs,
    )

    # Now start merging patterns
    merged_patterns = sorted(merged_patterns, key=lambda x: -len(x.seqlets))

    patterns = _filter_patterns(
        merged_patterns,
        min_seqlet_support=final_min_cluster_size,
        window_size=min_ic_windowsize,
        min_ic_in_window=min_ic_in_window,
        background=bg_freq,
        ppm_pseudocount=ppm_pseudocount,
    )

    # apply subclustering procedure on the final patterns
    for patternidx, pattern in enumerate(patterns):
        pattern.compute_subpatterns(
            subcluster_perplexity, n_seeds=n_leiden_runs, n_iterations=n_leiden_iterations
        )

    return patterns


def TFMoDISco(
    one_hot,
    hypothetical_contribs,
    sliding_window_size=21,
    flank_size=10,
    min_metacluster_size=100,
    weak_threshold_for_counting_sign=0.8,
    max_seqlets_per_metacluster=20000,
    target_seqlet_fdr=0.2,
    min_passing_windows_frac=0.03,
    max_passing_windows_frac=0.2,
    n_leiden_runs=50,
    n_leiden_iterations=-1,
    min_overlap_while_sliding=0.7,
    nearest_neighbors_to_compute=500,
    affmat_correlation_threshold=0.15,
    tsne_perplexity=10.0,
    frac_support_to_trim_to=0.2,
    min_num_to_trim_to=30,
    trim_to_window_size=20,
    initial_flank_to_add=5,
    prob_and_pertrack_sim_merge_thresholds=[(0.8, 0.8), (0.5, 0.85), (0.2, 0.9)],
    prob_and_pertrack_sim_dealbreaker_thresholds=[(0.4, 0.75), (0.2, 0.8), (0.1, 0.85), (0.0, 0.9)],
    subcluster_perplexity=50,
    merging_max_seqlets_subsample=300,
    final_min_cluster_size=20,
    min_ic_in_window=0.6,
    min_ic_windowsize=6,
    ppm_pseudocount=0.001,
    leiden_resolution=1.0,
    leiden_solver="modularity",
    verbose=False,
):
    contrib_scores = np.multiply(one_hot, hypothetical_contribs)

    track_set = core.TrackSet(
        one_hot=one_hot, contrib_scores=contrib_scores, hypothetical_contribs=hypothetical_contribs
    )

    seqlet_coords, threshold = extract_seqlets.extract_seqlets(
        attribution_scores=contrib_scores.sum(axis=2),
        window_size=sliding_window_size,
        flank=flank_size,
        suppress=(int(0.5 * sliding_window_size) + flank_size),
        target_fdr=target_seqlet_fdr,
        min_passing_windows_frac=min_passing_windows_frac,
        max_passing_windows_frac=max_passing_windows_frac,
        weak_threshold_for_counting_sign=weak_threshold_for_counting_sign,
    )

    seqlets = track_set.create_seqlets(seqlet_coords)

    pos_seqlets, neg_seqlets = [], []
    for seqlet in seqlets:
        flank = int(0.5 * (len(seqlet) - sliding_window_size))
        attr = np.sum(seqlet.contrib_scores[flank:-flank])

        if attr > threshold:
            pos_seqlets.append(seqlet)
        elif attr < -threshold:
            neg_seqlets.append(seqlet)

    del seqlets

    if len(pos_seqlets) > min_metacluster_size:
        pos_seqlets = pos_seqlets[:max_seqlets_per_metacluster]
        if verbose:
            print("Using {} positive seqlets".format(len(pos_seqlets)))

        pos_patterns = seqlets_to_patterns(
            seqlets=pos_seqlets,
            track_set=track_set,
            track_signs=1,
            min_overlap_while_sliding=min_overlap_while_sliding,
            nearest_neighbors_to_compute=nearest_neighbors_to_compute,
            affmat_correlation_threshold=affmat_correlation_threshold,
            tsne_perplexity=tsne_perplexity,
            n_leiden_iterations=n_leiden_iterations,
            n_leiden_runs=n_leiden_runs,
            frac_support_to_trim_to=frac_support_to_trim_to,
            min_num_to_trim_to=min_num_to_trim_to,
            trim_to_window_size=trim_to_window_size,
            initial_flank_to_add=initial_flank_to_add,
            prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
            prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
            subcluster_perplexity=subcluster_perplexity,
            merging_max_seqlets_subsample=merging_max_seqlets_subsample,
            final_min_cluster_size=final_min_cluster_size,
            min_ic_in_window=min_ic_in_window,
            min_ic_windowsize=min_ic_windowsize,
            ppm_pseudocount=ppm_pseudocount,
            leiden_resolution=leiden_resolution,
            leiden_solver=leiden_solver,
        )
    else:
        pos_patterns = None

    if len(neg_seqlets) > min_metacluster_size:
        neg_seqlets = neg_seqlets[:max_seqlets_per_metacluster]
        if verbose:
            print("Extracted {} negative seqlets".format(len(neg_seqlets)))

        neg_patterns = seqlets_to_patterns(
            seqlets=neg_seqlets,
            track_set=track_set,
            track_signs=-1,
            min_overlap_while_sliding=min_overlap_while_sliding,
            nearest_neighbors_to_compute=nearest_neighbors_to_compute,
            affmat_correlation_threshold=affmat_correlation_threshold,
            tsne_perplexity=tsne_perplexity,
            n_leiden_iterations=n_leiden_iterations,
            n_leiden_runs=n_leiden_runs,
            frac_support_to_trim_to=frac_support_to_trim_to,
            min_num_to_trim_to=min_num_to_trim_to,
            trim_to_window_size=trim_to_window_size,
            initial_flank_to_add=initial_flank_to_add,
            prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
            prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
            subcluster_perplexity=subcluster_perplexity,
            merging_max_seqlets_subsample=merging_max_seqlets_subsample,
            final_min_cluster_size=final_min_cluster_size,
            min_ic_in_window=min_ic_in_window,
            min_ic_windowsize=min_ic_windowsize,
            ppm_pseudocount=ppm_pseudocount,
            leiden_resolution=leiden_resolution,
            leiden_solver=leiden_solver,
        )
    else:
        neg_patterns = None

    return pos_patterns, neg_patterns


def main():
    desc = """TF-MoDISco is a motif detection algorithm that takes in nucleotide
        sequence and the attributions from a neural network model and return motifs
        that are repeatedly enriched for attriution score across the examples.
        This tool will take in one-hot encoded sequence, the corresponding
        attribution scores, and a few other parameters, and return the motifs."""

    # Read in the arguments
    parser = argparse.ArgumentParser(description=desc)
    subparsers = parser.add_subparsers(
        help="Must be either 'motifs', 'report', 'convert', 'convert-backward', 'meme', 'seqlet-bed', or 'seqlet-fasta'.",
        required=True,
        dest="cmd",
    )

    motifs_parser = subparsers.add_parser(
        "motifs",
        help="Run TF-MoDISco and extract the motifs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    motifs_parser.add_argument(
        "-s",
        "--sequences",
        type=str,
        help="A .npy or .npz file containing the one-hot encoded sequences.",
    )
    motifs_parser.add_argument(
        "-a",
        "--attributions",
        type=str,
        help="A .npy or .npz file containing the hypothetical attributions, i.e., the attributions for all nucleotides at all positions.",
    )
    motifs_parser.add_argument(
        "-i",
        "--h5py",
        type=str,
        help="A legacy h5py file containing the one-hot encoded sequences and shap scores.",
    )
    motifs_parser.add_argument(
        "-n",
        "--max_seqlets",
        type=int,
        required=True,
        help="The maximum number of seqlets per metacluster.",
    )
    motifs_parser.add_argument(
        "-l",
        "--n_leiden",
        type=int,
        default=2,
        help="The number of Leiden clusterings to perform with different random seeds.",
    )
    motifs_parser.add_argument(
        "-r",
        "--resolution",
        type=float,
        default=1.0,
    )
    motifs_parser.add_argument(
        "-e",
        "--solver",
        type=str,
        default="modularity",
    )
    motifs_parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=400,
        help="The window surrounding the peak center that will be considered for motif discovery.",
    )
    motifs_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="modisco_results.h5",
        help="The path to the output file.",
    )
    motifs_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Controls the amount of output from the code.",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Create a HTML report of the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    report_parser.add_argument(
        "-i",
        "--h5py",
        type=str,
        required=True,
        help="An HDF5 file containing the output from modiscolite.",
    )
    report_parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="A directory to put the output results including the html report.",
    )
    report_parser.add_argument(
        "-t",
        "--write-tomtom",
        action="store_true",
        default=False,
        help="Write the TOMTOM results to the output directory if flag is given.",
    )
    report_parser.add_argument(
        "-s",
        "--suffix",
        type=str,
        default="./",
        help="The suffix to add to the beginning of images. Should be equal to the output if using a Jupyter notebook.",
    )
    report_parser.add_argument(
        "-m", "--meme_db", type=str, default=None, help="A MEME file containing motifs."
    )
    report_parser.add_argument(
        "-n",
        "--n_matches",
        type=int,
        default=3,
        help="The number of top TOMTOM matches to include in the report.",
    )

    convert_parser = subparsers.add_parser("convert", help="Convert an old h5py to the new format.")
    convert_parser.add_argument(
        "-i", "--h5py", type=str, required=True, help="An HDF5 file formatted in the original way."
    )
    convert_parser.add_argument(
        "-o", "--output", type=str, required=True, help="An HDF5 file formatted in the new way."
    )

    convertback_parser = subparsers.add_parser(
        "convert-backward", help="Convert a new h5py to the old format."
    )
    convertback_parser.add_argument(
        "-i", "--h5py", type=str, required=True, help="An HDF5 file formatted in the new way."
    )
    convertback_parser.add_argument(
        "-o", "--output", type=str, required=True, help="An HDF5 file formatted in the old way."
    )

    meme_parser = subparsers.add_parser(
        "meme",
        help="""Output a MEME file from a
    modisco results file to stdout (default) and/or to a file (if specified).""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    meme_parser.add_argument(
        "-i", "--h5py", type=str, help="An HDF5 file containing the output from modiscolite."
    )
    meme_parser.add_argument(
        "-t",
        "--datatype",
        type=modiscolite.util.MemeDataType,
        choices=list(modiscolite.util.MemeDataType),
        required=True,
        help="""A case-sensitive string specifying the desired data of the output file.,
    The options are as follows:
    - 'PFM':      The position-frequency matrix.
    - 'CWM':      The contribution-weight matrix.
    - 'hCWM':     The hypothetical contribution-weight matrix; hypothetical
                  contribution scores are the contributions of nucleotides not encoded
                  by the one-hot encoding sequence.
    - 'CWM-PFM':  The softmax of the contribution-weight matrix.
    - 'hCWM-PFM': The softmax of the hypothetical contribution-weight matrix.""",
    )
    meme_parser.add_argument("-o", "--output", type=str, help="The path to the output file.")
    meme_parser.add_argument(
        "-q", "--quiet", action="store_true", default=False, help="Suppress output to stdout."
    )

    seqlet_bed_parser = subparsers.add_parser(
        "seqlet-bed",
        help="""Output a BED
    file of seqlets from a modisco results file to stdout (default) and/or to a
    file (if specified).""",
    )
    seqlet_bed_parser.add_argument(
        "-i",
        "--h5py",
        type=str,
        required=True,
        help="An HDF5 file containing the output from modiscolite.",
    )
    seqlet_bed_parser.add_argument(
        "-o", "--output", type=str, default=None, help="The path to the output file."
    )
    seqlet_bed_parser.add_argument(
        "-p",
        "--peaksfile",
        type=str,
        required=True,
        help="The path to the peaks file. This is to compute the absolute start and\
    end positions of the seqlets within a reference genome, as well as the chroms.",
    )
    seqlet_bed_parser.add_argument(
        "-c",
        "--chroms",
        type=str,
        required=True,
        help="""A comma-delimited list of chromosomes, or '*', denoting which
    chromosomes to process. Should be the same set of chromosomes used during
    interpretation. '*' will use every chr in the provided peaks file.
    Examples: 'chr1,chr2,chrX' || '*' || '1,2,X'.""",
    )
    seqlet_bed_parser.add_argument(
        "-q", "--quiet", action="store_true", default=False, help="Suppress output to stdout."
    )
    seqlet_bed_parser.add_argument(
        "-w",
        "--windowsize",
        type=int,
        help="""Optional. This is for backwards compatibility for older modisco h5
    files that don't contain the window size as an attribute. This should be set
    the to size of the window around the peak center that was used for.""",
    )

    seqlet_fasta_parser = subparsers.add_parser(
        "seqlet-fasta",
        help="""Output a FASTA
    file of seqlets from a modisco results file to stdout (default) and/or to a
    file (if specified).""",
    )
    seqlet_fasta_parser.add_argument(
        "-i",
        "--h5py",
        type=str,
        required=True,
        help="An HDF5 file containing the output from modiscolite.",
    )
    seqlet_fasta_parser.add_argument(
        "-o", "--output", type=str, default=None, help="The path to the output file."
    )
    seqlet_fasta_parser.add_argument(
        "-p",
        "--peaksfile",
        type=str,
        required=True,
        help="The path to the peaks file. This is to compute the absolute start and\
    end positions of the seqlets within a reference genome, as well as the chroms.",
    )
    seqlet_fasta_parser.add_argument(
        "-s",
        "--sequences",
        type=str,
        required=True,
        help="A .npy or .npz file containing the one-hot encoded sequences.",
    )
    seqlet_fasta_parser.add_argument(
        "-c",
        "--chroms",
        type=str,
        required=True,
        help="""A comma-delimited list of chromosomes, or '*', denoting which
    chromosomes to process. Should be the same set of chromosomes used during
    interpretation. '*' will use every chr in the provided peaks file.
    Examples: 'chr1,chr2,chrX' || '*' || '1,2,X'.""",
    )
    seqlet_fasta_parser.add_argument(
        "-q", "--quiet", action="store_true", default=False, help="Suppress output to stdout."
    )
    seqlet_fasta_parser.add_argument(
        "-w",
        "--windowsize",
        type=int,
        help="""Optional. This is for backwards compatibility for older modisco h5
    files that don't contain the window size as an attribute. This should be set
    the to size of the window around the peak center that was used for.""",
    )

    def convert_arg_chroms_to_list(chroms: str) -> Union[List[str], Literal["*"]]:
        """Converts the chroms argument to a list of chromosomes."""
        if chroms == "*":
            # Return all chromosome numbers
            return "*"
        else:
            return chroms.split(",")

    # Pull the arguments
    args = parser.parse_args()

    if args.cmd == "motifs":
        if args.h5py is not None:
            # Load the scores
            scores = h5py.File(args.h5py, "r")

            try:
                center = scores["hyp_scores"].shape[1] // 2
                start, end = calculate_window_offsets(center, args.window)

                attributions = scores["hyp_scores"][:, start:end, :]
                sequences = scores["input_seqs"][:, start:end, :]
            except KeyError:
                center = scores["shap"]["seq"].shape[2] // 2
                start, end = calculate_window_offsets(center, args.window)

                attributions = scores["shap"]["seq"][:, :, start:end].transpose(0, 2, 1)
                sequences = scores["raw"]["seq"][:, :, start:end].transpose(0, 2, 1)

            scores.close()

        else:
            if args.sequences[-3:] == "npy":
                sequences = np.load(args.sequences)
            elif args.sequences[-3:] == "npz":
                sequences = np.load(args.sequences)["arr_0"]

            if args.attributions[-3:] == "npy":
                attributions = np.load(args.attributions)
            elif args.attributions[-3:] == "npz":
                attributions = np.load(args.attributions)["arr_0"]

            center = sequences.shape[2] // 2
            start, end = calculate_window_offsets(center, args.window)

            sequences = sequences[:, :, start:end].transpose(0, 2, 1)
            attributions = attributions[:, :, start:end].transpose(0, 2, 1)

        if sequences.shape[1] < args.window:
            raise ValueError(
                "Window ({}) cannot be ".format(args.window)
                + "longer than the sequences".format(sequences.shape)
            )

        sequences = sequences.astype("float32")
        attributions = attributions.astype("float32")

        pos_patterns, neg_patterns = TFMoDISco(
            hypothetical_contribs=attributions,
            one_hot=sequences,
            max_seqlets_per_metacluster=args.max_seqlets,
            sliding_window_size=20,
            flank_size=5,
            target_seqlet_fdr=0.05,
            n_leiden_runs=args.n_leiden,
            leiden_resolution=args.resolution,
            leiden_solver=args.solver,
            verbose=args.verbose,
        )

        modiscolite.io.save_hdf5(args.output, pos_patterns, neg_patterns, args.window)

    elif args.cmd == "meme":
        modiscolite.io.write_meme_from_h5(args.h5py, args.datatype, args.output, args.quiet)

    elif args.cmd == "seqlet-bed":
        modiscolite.io.write_bed_from_h5(
            args.h5py,
            args.peaksfile,
            args.output,
            convert_arg_chroms_to_list(args.chroms),
            args.windowsize,
            args.quiet,
        )

    elif args.cmd == "seqlet-fasta":
        modiscolite.io.write_fasta_from_h5(
            args.h5py,
            args.peaksfile,
            args.sequences,
            args.output,
            convert_arg_chroms_to_list(args.chroms),
            args.windowsize,
            args.quiet,
        )

    elif args.cmd == "report":
        modiscolite.report.report_motifs(
            args.h5py,
            args.output,
            img_path_suffix=args.suffix,
            meme_motif_db=args.meme_db,
            is_writing_tomtom_matrix=args.write_tomtom,
            top_n_matches=args.n_matches,
        )

    elif args.cmd == "convert":
        modiscolite.io.convert(args.h5py, args.output)

    elif args.cmd == "convert-backward":
        modiscolite.io.convert_new_to_old(args.h5py, args.output)


if __name__ == "__main__":
    main()
