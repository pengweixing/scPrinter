# from train import train_model
# from visualization_helper import predict_footprints
from pathlib import Path

import numpy as np

from ..genome import Genome
from ..io import scPrinter


def seq_model_config(
    printer: scPrinter,
    peak_file: str | Path,
    cell_grouping: list[list[str]] | list[str] | np.ndarray,
    group_names: list[str] | str,
    genome: Genome,
    fold: int,
    overwrite_bigwig=True,
    model_name=None,
    model_configs={},
):
    """
    Generate a configuration dictionary for the seq2PRINT model

    Parameters:
    - printer (scPrinter): The scPrinter object containing the insertion profile and other relevant information.
    - peak_file (str | Path): The file path to the peak file (from scp.peak.clean_macs2 and saved as a bed text file).
    - cell_grouping (list[list[str]] | list[str] | np.ndarray): The cell grouping for creating the model configuration.
    - group_names (list[str] | str): The names of the groups.
    - genome (Genome): The genome object.
    - fold (int): The fold for splitting the data.
    - overwrite_bigwig (bool): Whether to overwrite existing bigwig files. Default is True.
    - model_name (str | None): The name of the model. Default is None.
    - model_configs (dict): Additional configurations for the model. Default is an empty dictionary.

    Returns:
    - template_json (dict): The configuration dictionary for the sequence model.
    """

    if type(group_names) not in [np.ndarray, list]:
        group_names = [group_names]
        cell_grouping = [cell_grouping]
    if len(group_names) > 1:
        raise NotImplementedError("Currently only support one group at a time")
    if model_name is None and "savename" not in model_configs:
        raise ValueError("Please provide a model name or a savename in model_configs")
    if "group_bigwig" not in printer.insertion_file.uns:
        printer.insertion_file.uns["group_bigwig"] = {}

    for name, grouping in zip(group_names, cell_grouping):
        if name in printer.insertion_file.uns["group_bigwig"]:
            if not overwrite_bigwig:
                print("bigwig for %s already exists, skip" % name)
                continue
        sync_footprints(printer, grouping, name)

    template_json = {
        "peaks": peak_file,
        "signals": [printer.insertion_file.uns["group_bigwig"][name] for name in group_names],
        "genome": genome.name,
        "split": genome.splits[fold],
        "n_filters": 1024,
        "bottleneck_factor": 1,
        "no_inception": False,
        "groups": 8,
        "n_layers": 8,
        "n_inception_layers": 8,
        "inception_layers_after": True,
        "inception_version": 2,
        "activation": "gelu",
        "batch_norm_momentum": 0.1,
        "dilation_base": 1,
        "rezero": False,
        "batch_norm": True,
        "batch_size": 64,
        "head_kernel_size": 1,
        "kernel_size": 3,
        "weight_decay": 1e-3,
        "lr": 3e-3,
        "scheduler": False,
        "savename": model_name,
        "amp": True,
        "ema": True,
    }
    for key in model_configs:
        template_json[key] = model_configs[key]

    return template_json
