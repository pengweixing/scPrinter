import numpy
import torch


def DNA_one_hot(sequence, alphabet="ACGT", dtype="float", device="cpu"):
    """Convert a DNA sequence string into a one-hot encoding tensor.
    Parameters
    ----------
    sequence : str
        DNA sequence string.
    dtype : str
        Data type of the returned tensor.
    device : str
        Device of the returned tensor.
    verbose : bool
        If True, print progress.

    Returns
    -------
    torch.Tensor
        One-hot encoding tensor.
    """
    lookup = {char: i for i, char in enumerate(alphabet)}
    lookup["N"] = -1
    embed = torch.zeros((len(alphabet) + 1, len(sequence)), dtype=torch.int8, device=device)
    embed[[lookup[char] for char in sequence], torch.arange(len(sequence))] = 1

    return embed[:-1, :]
