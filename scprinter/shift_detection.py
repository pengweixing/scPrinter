import gzip
import os

import numpy as np
import pandas as pd
from pyfaidx import Fasta


def one_hot_encode(seq):
    """
    One-hot encode a genomic sequence.
    :param seq: A string representing a genomic sequence.
    :return: A len_seq x 4 numpy array. Columns are A, C, G, T.
    """

    # Split the sequence into a list of characters
    seq = list(seq)

    # Map each base to 0-4 integers
    encode_dict = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    base_inds = np.array(list(map(encode_dict.get, list(seq))))

    # Encode as one-hot matrix
    encoding = np.zeros((len(seq), 5))
    encoding[np.arange(len(seq)), base_inds] = 1

    # Remove Ns
    encoding = encoding[:, :4]

    return encoding


def get_bias(beds, genome, context_radius=20):
    genome_seq = Fasta(genome.fetch_fa())
    forward_bias = []
    reverse_bias = []
    for frags in np.array(beds):
        chrom, start, end = frags[0], int(frags[1]), int(frags[2])
        forward_context = one_hot_encode(
            genome_seq[chrom][start - context_radius : start + context_radius].seq.upper()
        )
        reverse_context = one_hot_encode(
            genome_seq[chrom][end - context_radius : end + context_radius].seq.upper()
        )
        forward_bias.append(forward_context)
        reverse_bias.append(reverse_context)

    forward_bias = np.mean(np.array(forward_bias), axis=0)
    reverse_bias = np.mean(np.array(reverse_bias), axis=0)

    return forward_bias, reverse_bias


def plot_bias(bias_pfm):
    """
    Convert the position frequency matrix to information content matrix and plot as a sequence logo.
    :param bias_pfm: Position frequency matrix. Rows are positions, columns are A, C, G, T.
    :return: No return value. Plot will be visualized
    """

    import logomaker

    # Get Position-Probability-Matrix (PPM)
    bias_ppm = bias_pfm / np.sum(bias_pfm, axis=1)[:, None]

    # Comput information content matrix
    # For details, see instructions here https://www.bioconductor.org/packages/devel/bioc/vignettes/universalmotif/inst/doc/IntroductionToSequenceMotifs.pdf
    uncertainty = -np.sum(bias_ppm * np.log2(bias_ppm), axis=1)
    information_content = 2 - uncertainty
    icm = pd.DataFrame(bias_ppm * information_content[:, None], columns=["A", "C", "G", "T"])

    # Visulize results
    logomaker.Logo(icm, vpad=0.1, width=0.8)


def circular_shift(matrix, shift):
    """
    Circular shift a matrix by a given distance
    :param matrix: Matrix to be shifted
    :param shift: Shift distance
    :return: Shifted matrix
    """
    return np.concatenate((matrix[-shift:, :], matrix[:-shift, :]), axis=0)


def mse(matrix1, matrix2):
    """
    Compute mean squared error between two matrices
    :param matrix1: Input matrix 1
    :param matrix2: Input matrix 2
    :return: MSE valu
    """
    return np.mean((matrix1 - matrix2) ** 2)


def circular_detect(ref_bias, query_bias):
    """
    Detect shift distance between two bias matrices
    :param ref_bias: Reference bias matrix
    :param query_bias: Query bias matrix
    :return: Shift distance
    """

    # Make sure two matrices are the same shape
    assert np.shape(ref_bias) == np.shape(query_bias), "Input matrices must have the same shape"

    # Circular shift the query bias matrix and compute MSE at each shift distance
    context_radius = int(np.shape(ref_bias)[0] / 2)
    shift_mse = [
        mse(circular_shift(ref_bias, shift), query_bias)
        for shift in range(-context_radius, context_radius)
    ]

    # Return the shift distance that minimizes MSE
    return np.argmin(shift_mse) - context_radius


def count_lines(filename):
    if ".gz" in filename:
        with gzip.open(filename, "rt") as file:
            return sum(1 for line in file)
    else:
        with open(filename, "r") as file:
            return sum(1 for line in file)


def sample_bed_file_to_dataframe(filename, sample_size):
    import random

    line_count = count_lines(filename)
    sample_indices = set(random.sample(range(line_count), sample_size))

    samples = []
    if ".gz" in filename:
        file = gzip.open(filename, "rt")
    else:
        file = open(filename, "r")
    for i, line in enumerate(file):
        if i in sample_indices:
            samples.append(line.strip().split("\t"))
            if len(samples) == sample_size:
                break

    file.close()
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    return df


def detect_shift(frags, genome):
    ref_forward_bias, ref_reverse_bias = np.array(
        [
            [0.22801, 0.267627, 0.270118, 0.234245],
            [0.230582, 0.263092, 0.266591, 0.239735],
            [0.230966, 0.263395, 0.277853, 0.227786],
            [0.237373, 0.254811, 0.268621, 0.239195],
            [0.227756, 0.262171, 0.281433, 0.22864],
            [0.229343, 0.267518, 0.266358, 0.236781],
            [0.233688, 0.261873, 0.268756, 0.235683],
            [0.233246, 0.258417, 0.267473, 0.240864],
            [0.233571, 0.250868, 0.278438, 0.237123],
            [0.210302, 0.265267, 0.273532, 0.250899],
            [0.197471, 0.270346, 0.276262, 0.255921],
            [0.239908, 0.224086, 0.353718, 0.182288],
            [0.209381, 0.325079, 0.290731, 0.174809],
            [0.27844, 0.226113, 0.273372, 0.222075],
            [0.174853, 0.40936, 0.237924, 0.177863],
            [0.345859, 0.149262, 0.228952, 0.275927],
            [0.165938, 0.270514, 0.480596, 0.082952],
            [0.180195, 0.29281, 0.221314, 0.305681],
            [0.179562, 0.332162, 0.237392, 0.250884],
            [0.125249, 0.409938, 0.152056, 0.312757],
            [0.334499, 0.169781, 0.172222, 0.323498],
            [0.300765, 0.151369, 0.411777, 0.136089],
            [0.255409, 0.225268, 0.345623, 0.1737],
            [0.294189, 0.221911, 0.303888, 0.180012],
            [0.125051, 0.422991, 0.270467, 0.181491],
            [0.25149, 0.249151, 0.18284, 0.316519],
            [0.190253, 0.229807, 0.393018, 0.186922],
            [0.222202, 0.265248, 0.242275, 0.270275],
            [0.171303, 0.27846, 0.331291, 0.218946],
            [0.182575, 0.338698, 0.238956, 0.239771],
            [0.249742, 0.270574, 0.275662, 0.204022],
            [0.242147, 0.26794, 0.272813, 0.2171],
            [0.238624, 0.26772, 0.261579, 0.232077],
            [0.234333, 0.259208, 0.264034, 0.242425],
            [0.234719, 0.257802, 0.269948, 0.237531],
            [0.232542, 0.259099, 0.274311, 0.234048],
            [0.228187, 0.266636, 0.273797, 0.23138],
            [0.231669, 0.263802, 0.26339, 0.241139],
            [0.228852, 0.26637, 0.274964, 0.229814],
            [0.23085, 0.260809, 0.266385, 0.241956],
        ]
    ), np.array(
        [
            [0.227808, 0.264997, 0.272057, 0.235138],
            [0.229786, 0.261645, 0.267999, 0.24057],
            [0.229542, 0.265141, 0.277305, 0.228012],
            [0.2364, 0.25577, 0.267527, 0.240303],
            [0.228397, 0.261634, 0.278268, 0.231701],
            [0.227913, 0.266164, 0.267063, 0.23886],
            [0.233518, 0.263004, 0.268147, 0.235331],
            [0.231852, 0.259135, 0.267172, 0.241841],
            [0.233683, 0.251586, 0.27724, 0.237491],
            [0.209089, 0.266759, 0.272593, 0.251559],
            [0.200028, 0.266882, 0.283073, 0.250017],
            [0.235364, 0.228948, 0.348828, 0.18686],
            [0.213109, 0.320979, 0.290091, 0.175821],
            [0.271552, 0.238546, 0.267749, 0.222153],
            [0.186768, 0.377292, 0.242154, 0.193786],
            [0.312909, 0.176144, 0.256335, 0.254612],
            [0.176285, 0.258176, 0.440994, 0.124545],
            [0.174572, 0.297044, 0.226605, 0.301779],
            [0.174146, 0.338917, 0.232244, 0.254693],
            [0.129944, 0.406257, 0.1538, 0.309999],
            [0.329279, 0.166597, 0.17534, 0.328784],
            [0.301616, 0.149981, 0.416916, 0.131487],
            [0.251985, 0.230058, 0.337205, 0.180752],
            [0.300088, 0.215177, 0.299529, 0.185206],
            [0.083786, 0.462517, 0.282588, 0.171109],
            [0.274189, 0.22066, 0.155771, 0.34938],
            [0.174265, 0.225577, 0.425875, 0.174283],
            [0.219996, 0.271626, 0.230628, 0.27775],
            [0.171276, 0.28058, 0.333658, 0.214486],
            [0.178319, 0.342787, 0.233408, 0.245486],
            [0.255848, 0.265675, 0.27829, 0.200187],
            [0.241528, 0.267603, 0.273153, 0.217716],
            [0.238694, 0.266755, 0.261775, 0.232776],
            [0.234914, 0.2593, 0.26413, 0.241656],
            [0.234639, 0.258018, 0.270217, 0.237126],
            [0.231615, 0.258526, 0.275489, 0.23437],
            [0.226413, 0.269257, 0.273205, 0.231125],
            [0.230697, 0.264718, 0.261399, 0.243186],
            [0.228284, 0.265556, 0.273413, 0.232747],
            [0.230259, 0.260566, 0.268176, 0.240999],
        ]
    )
    if os.path.getsize(frags) / (1024**3) < 2:
        frags = pd.read_csv(frags, sep="\t", header=None).sample(10000)
    else:
        frags = sample_bed_file_to_dataframe(frags, 10000)
    forward_bias, reverse_bias = get_bias(frags, genome)
    return 4 - circular_detect(ref_forward_bias, forward_bias), -5 - circular_detect(
        ref_reverse_bias, reverse_bias
    )
