from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm.auto import *

try:
    from scprinter.genome import Genome
    from scprinter.motifs import Motifs
    from scprinter.seq.Models import *
    from scprinter.utils import DNA_one_hot, regionparser
    from scprinter.utils import zscore2pval_torch as z2p
except ImportError:
    from ...genome import Genome
    from ...motifs import Motifs
    from ...utils import DNA_one_hot, regionparser
    from ...utils import zscore2pval_torch as z2p
    from ..Models import *

verbose_template = "Please copy the following command in your terminal and run it to start the job"
launch_template = "Launching the following command now (no action needed from your side)"

torch.set_num_threads(4)


# Function to get model outputs in batches and concatenate them separately
def get_model_outputs(models, seq_list, batch_size=64, device="cuda"):
    all_outputs1 = []
    all_outputs2 = []

    for seq in seq_list:
        seq = seq  # Move input to GPU
        outputs1 = []
        outputs2 = []
        with torch.autocast(
            device_type="cuda" if "cuda" in device else "cpu",
            dtype=torch.bfloat16,
            enabled=device != "cpu",
        ):
            with torch.no_grad():  # Disable gradient calculation
                for i in range(0, len(seq), batch_size):
                    batch = seq[i : i + batch_size].to(device).float()
                    output1, output2 = 0, 0
                    for model in models:
                        a, b = model(batch)
                        output1 += a
                        output2 += b
                    outputs1.append(z2p(output1).detach().cpu())  # Move output1 to CPU and collect
                    outputs2.append(z2p(output2).detach().cpu())  # Move output2 to CPU and collect
            all_outputs1.append(torch.cat(outputs1))
            all_outputs2.append(torch.cat(outputs2))
    return all_outputs1, all_outputs2


def get_delta_effects(
    model_path: str | Path | list[str | Path],
    genome: Genome,
    region_path: str,
    motifs: Motifs | str,
    motif_sample_mode: Literal["argmax", "multinomial"] = "argmax",
    lora_ids: list | None = None,
    prefix: str = "",
    sample_num=25000,
    gpus: int | list[int] | None = None,
    collapse_footprint_across_bins=False,
    random_seq=False,
    launch=True,
    verbose=False,
    save_path=None,
):
    """
    Calculate the delta effects of the motifs on the model

    Parameters
    ----------
    xxx


    Returns
    -------

    """
    if type(model_path) is not list:
        model_path = [model_path]

    motif2dump = []

    if isinstance(motifs, str):
        with h5py.File(motifs, "r") as f:
            for group in ["pos_patterns", "neg_patterns"]:
                for key in f[group].keys():
                    motif = f[group][key]["sequence"][:].T
                    nn = f"{prefix}_{group}.{key}"
                    motif2dump.append(motif)
    elif isinstance(motifs, Motifs):
        for motif in motifs.all_motifs:
            pfm = motif.counts
            pfm = np.array([pfm[key] for key in ["A", "C", "G", "T"]])
            motif2dump.append(pfm)
    else:
        motif2dump = list(motifs.values())

    # Each motif has 3 dim to support composite motif (sth like left motif and right motif)

    motif2dump = np.array(motif2dump, dtype="object")

    motif_temp_file = os.path.join(save_path, "motif.npy")
    np.save(motif_temp_file, motif2dump)
    delta_effects_tempfile = os.path.join(save_path, "delta_effects.npy")
    command = (
        ["seq2print_delta", "--pt"]
        + model_path
        + (["--ids", ",".join(map(str, lora_ids))] if lora_ids is not None else [])
        + [
            "--genome",
            genome.name,
            "--peaks",
            region_path,
            "--motifs",
            motif_temp_file,
            "--mode",
            motif_sample_mode,
            "--gpus",
        ]
        + [str(x) for x in gpus]
        + ["--sample_num", str(sample_num), "--output", delta_effects_tempfile]
        + (["--collapse_bins"] if collapse_footprint_across_bins else [])
        + (["--random_seq"] if random_seq else [])
        + (["--verbose"] if verbose else [])
    )
    command_str = " ".join(command)
    if verbose:
        if launch:
            print(launch_template)
        else:
            print(verbose_template)
    print(command_str)
    if launch:
        subprocess.run(command)
        return np.load(delta_effects_tempfile, allow_pickle=True)


# def get_delta_effects(
#     models: torch.nn.Module | str | list[str] | list[torch.nn.Module],
#     genome: Genome,
#     regions: str,
#     motifs: Motifs | str,
#     prefix: str = "",
#     sample_num=25000,
#     batch_size=256,
#     device="cuda",
# ):
#     """
#     Calculate the delta effects of the motifs on the model
#
#     Parameters
#     ----------
#     model: torch.nn.Module | str
#         The seq2PRINT model to be evaluated
#     regions:
#     motifs
#     sample_num
#
#     Returns
#     -------
#
#     """
#     if type(models) is not list:
#         models = [models]
#
#     CREs = regionparser(regions, None)
#
#     name2concensus = {}
#     if isinstance(motifs, str):
#         with h5py.File(motifs, "r") as f:
#             for group in ["pos_patterns", "neg_patterns"]:
#                 for key in f[group].keys():
#                     motif = f[group][key]["sequence"][:].T
#                     nucleotide = np.array(list("ACGT"))
#                     concensus = nucleotide[np.argmax(motif, axis=0)]
#                     concensus = "".join(list(concensus))
#                     nn = f"{prefix}_{group}.{key}"
#                     name2concensus[nn] = concensus
#     else:
#         for motif in motifs.all_motifs:
#             pfm = motif.counts
#             pfm = np.array([pfm[key] for key in ["A", "C", "G", "T"]])
#             nucleotide = np.array(list("ACGT"))
#             concensus = nucleotide[np.argmax(pfm, axis=0)]
#             concensus = "".join(list(concensus))
#             name2concensus[motif.name] = concensus
#     name2delta = {}
#     for model_id, model in enumerate(models):
#         if isinstance(model, str):
#             model = torch.load(model, map_location="cpu")
#         model = model.to(device)
#
#         bar = tqdm(name2concensus, desc="motifs")
#         for name in bar:
#             bar.set_description(f"working on {name} with model {model_id}")
#             concensus = name2concensus[name]
#
#             seq_motif = []
#             seq_baseline = []
#             for i, region in enumerate(np.array(CREs.sample(sample_num))):
#                 chr, start, end = region[0], region[1], region[2]
#
#                 center = int((start + end) // 2)
#                 pad = 1840 // 2
#                 seq = genome.fetch_seq(chr, center - pad, center + pad)
#
#                 seq_baseline.append(DNA_one_hot(seq))
#
#                 seq = list(seq)
#                 pad = (len(seq) - len(concensus)) // 2
#                 seq[pad : pad + len(concensus)] = list(concensus)
#                 seq_motif.append(DNA_one_hot(seq))
#
#             seq_baseline = torch.stack(seq_baseline, axis=0)
#             seq_motif = torch.stack(seq_motif, axis=0)
#
#             # List of input sequences
#             input_sequences = [seq_baseline, seq_motif]
#
#             # Get model outputs for all sequences
#             outputs1, outputs2 = get_model_outputs([model], input_sequences, batch_size, device)
#             # Now, concatenated_outputs1 and concatenated_outputs2 hold the concatenated results of the first and second outputs of the model
#             # delta = z2p(outputs1[1]) - z2p(outputs1[0])
#             delta = outputs1[1] - outputs1[0]
#             if name not in name2delta:
#                 name2delta[name] = []
#             name2delta[name].append(delta.mean(axis=0).float().detach().cpu().numpy())
#     for name in name2delta:
#         name2delta[name] = np.mean(name2delta[name], axis=0)
#     return name2delta


def plot_delta_effects(name2delta, flank=200, vmin=-0.3, vmax=0.3, save_path=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name in name2delta:
        delta = name2delta[name]
        pad = delta.shape[1] // 2 - flank
        if pad > 0:
            delta = delta[:, pad:-pad]
        plt.figure(figsize=(4, 2))
        sns.heatmap(delta[::-1] / 5, cmap="RdBu_r", vmin=vmin, vmax=vmax, cbar=False)
        plt.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(os.path.join(save_path, f"{name}.png"))
        plt.close("all")
