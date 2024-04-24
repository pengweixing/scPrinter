import argparse
import gc
import json
import math
import os
import os.path
import subprocess
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import h5py
import numpy as np
import pandas as pd
import torch

import scprinter as scp
from scprinter.seq.attribution_wrapper import *
from scprinter.seq.attributions import *
from scprinter.seq.dataloader import *
from scprinter.seq.Models import *

parser = argparse.ArgumentParser(description="Bindingscore BPNet")
parser.add_argument("--pt", type=str, default="config.JSON", help="model.pt")
parser.add_argument("--models", type=str, default=None, help="model")
parser.add_argument("--genome", type=str, default="hg38", help="genome")
parser.add_argument("--peaks", type=str, default=None, help="peaks")
parser.add_argument("--method", nargs="+", type=str, default=None, help="method")
parser.add_argument("--wrapper", nargs="+", type=str, default=None, help="method")
parser.add_argument("--nth_output", nargs="+", type=str, default=None, help="nth output")
parser.add_argument("--save_pred", action="store_true", default=False, help="save pred")
parser.add_argument("--gpus", nargs="+", type=int, help="gpus")
parser.add_argument("--overwrite", action="store_true", default=False, help="overwrite")
parser.add_argument("--replicate", type=int, default=None, help="replicate")
parser.add_argument("--decay", type=float, default=None, help="decay")
parser.add_argument("--disable_ema", action="store_true", default=False, help="disable emas")

decay = None
torch.set_num_threads(4)
args = parser.parse_args()
ids = args.models
ids = ids.split(",")
ids = [int(i) for i in ids]
gpus = args.gpus
wrappers = args.wrapper
methods = args.method
nth_output = args.nth_output
print(gpus)
if len(gpus) == 1:
    torch.cuda.set_device(int(gpus[0]))


def load_entire_hdf5(dct):
    if isinstance(dct, h5py.Dataset):
        return dct[()]
    ret = {}
    for k, v in dct.items():
        ret[k] = load_entire_hdf5(v)
    return ret


def loadDispModel(h5Path):
    with h5py.File(h5Path, "r") as a:
        dispmodels = load_entire_hdf5(a)
        for model in dispmodels:
            dispmodels[model]["modelWeights"] = [
                torch.from_numpy(dispmodels[model]["modelWeights"][key]).float()
                for key in ["ELT1", "ELT2", "ELT3", "ELT4"]
            ]
    return dispmodels


split = {
    "test": ["chr1", "chr3", "chr6"]
    + ["chr8", "chr20"]
    + [
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
        "chrY",
    ]
}


peaks = args.peaks
summits = pd.read_table(peaks, sep="\t", header=None)
summits = summits.drop_duplicates([0, 1, 2])  # drop exact same loci
summits["summits"] = (summits[1] + summits[2]) // 2
print(summits)
summits = summits[[0, "summits"]]
summits["summits"] = np.array(summits["summits"], dtype=int)


acc_model = torch.load(args.pt, map_location="cpu").cuda()
# set coverage to be the same
mm = acc_model.coverages.weight.data[:, -1].mean()
acc_model.coverages.weight.data[:, -1] = mm
acc_model.eval()
dna_len = acc_model.dna_len

print(acc_model.profile_cnn_model.coverages.weight)

signal_window = 1000
print("signal_window", signal_window, "dna_len", dna_len)
genome = args.genome
if genome == "hg38":
    genome = scp.genome.hg38
elif genome == "mm10":
    genome = scp.genome.mm10
else:
    raise ValueError("genome not supported")
bias = str(genome.fetch_bias())[:-3] + ".bw"
signals = [bias, bias]

datasets = {
    k: ChromBPDataset(
        signals=signals,
        ref_seq=genome.fetch_fa(),
        summits=summits[summits[0].isin(split[k])],
        DNA_window=dna_len,
        signal_window=signal_window,
        max_jitter=0,
        min_counts=None,
        max_counts=None,
        cached=False,
        reverse_compliment=False,
        device="cpu",
    )
    for k in split
}

dataloader = {
    k: ChromBPDataLoader(
        dataset=datasets[k],
        batch_size=64,
        num_workers=0,
        pin_memory=True,
        shuffle=True if k in ["train"] else False,
    )
    for k in split
}

summits = dataloader["test"].dataset.summits
regions = np.array([summits[:, 0], summits[:, 1] - dna_len // 2, summits[:, 1] + dna_len // 2]).T
acc_model.upsample = False
if len(gpus) > 1:
    print(acc_model)

# nth_output = torch.as_tensor([5, 10]) - 1
params_combination = []
save_dir = f"{args.pt}_deepshap"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for wrapper in wrappers:
    for n_out in nth_output:
        for method in methods:
            extra = f"_{n_out}_"
            if len(gpus) > 1:
                # datasets['test'].cache()
                n_split = len(gpus)

                unfinished_ids = []
                for id in ids:
                    if os.path.exists(
                        os.path.join(
                            save_dir, f"model_{id}.hypo.{wrapper}.{method}{extra}.{decay}.npz"
                        )
                    ) and (not args.overwrite):
                        print("exists")
                        continue
                    else:
                        unfinished_ids.append(id)
                unfinished_ids = np.array(unfinished_ids)
                bs = int(math.ceil(len(unfinished_ids) / n_split))
                ids_batch = np.array_split(unfinished_ids, n_split)

                commands = [
                    [
                        "python",
                        "evaluation_lora_model_v2.py",
                        "--pt",
                        args.pt,
                        "--genome",
                        args.genome,
                        "--peaks",
                        args.peaks,
                        "--model",
                        ",".join([str(i) for i in id_batch]),
                        "--method",
                        method,
                        "--wrapper",
                        wrapper,
                        "--nth_output",
                        str(n_out),
                        "--gpus",
                        str(gpu),
                    ]
                    + (["--overwrite"] if args.overwrite else [])
                    for i, (id_batch, gpu) in enumerate(zip(ids_batch, gpus))
                ]
                print(commands)
                pool = ProcessPoolExecutor(max_workers=len(gpus))
                for gpu, command in zip(gpus, commands):
                    pool.submit(subprocess.run, command)

                pool.shutdown(wait=True)
            else:
                for id in ids:
                    print("working on ", id, wrapper, method, n_out, decay)
                    if os.path.exists(
                        os.path.join(
                            save_dir, f"model_{id}.hypo.{wrapper}.{method}{extra}.{decay}.npz"
                        )
                    ) and (not args.overwrite):
                        print("exists")
                        continue
                    model_0 = acc_model.collapse(int(id))
                    model_0 = model_0.cuda()
                    model_0.eval()

                    if type(n_out) is not torch.Tensor:
                        if "," in n_out:
                            n_out = n_out.split(",")
                            n_out = torch.as_tensor([int(i) for i in n_out])
                        elif "-" in n_out:
                            n_out_start, n_out_end = n_out.split("-")
                            n_out = torch.as_tensor(
                                [i for i in range(int(n_out_start), int(n_out_end))]
                            )
                        else:
                            n_out = torch.as_tensor([int(n_out)])
                        if n_out[0] < 0:
                            n_out = None

                    if wrapper == "classification":
                        model = ProfileWrapperFootprintClass(
                            model_0, nth_output=n_out, res=1, reduce_mean=False, decay=decay
                        )
                    elif wrapper == "regression":
                        model = ProfileWrapperFootprint(
                            model_0, nth_output=n_out, res=1, reduce_mean=False
                        )
                    elif wrapper == "classification_reduce":
                        model = ProfileWrapperFootprintClass(
                            model_0, nth_output=n_out, res=1, reduce_mean=True, decay=decay
                        )
                    elif wrapper == "regression_reduce":
                        model = ProfileWrapperFootprint(
                            model_0, nth_output=n_out, res=1, reduce_mean=True
                        )
                    elif wrapper == "just_sum":
                        model = JustSumWrapper(model_0, nth_output=n_out, res=1, threshold=0.301)
                    elif wrapper == "count":
                        model = CountWrapper(model_0)
                    model = model.cuda()
                    print("method", method, wrapper)

                    datasets["test"].cache()
                    attributions = calculate_attributions(
                        model,
                        X=datasets["test"].cache_seqs,
                        n_shuffles=20,
                        method=method,
                        verbose=True,
                    )
                    projected_attributions = projected_shap(
                        attributions, datasets["test"].cache_seqs, 64, "cuda"
                    )
                    hypo, ohe = (
                        attributions.detach().cpu().numpy(),
                        datasets["test"].cache_seqs.detach().cpu().numpy(),
                    )

                    print(projected_attributions.shape, hypo.shape, ohe.shape, regions)
                    # vs = projected_attributions
                    # low, median, high = np.quantile(vs, 0.05), np.quantile(vs, 0.5), np.quantile(vs, 0.95)
                    vs = projected_attributions[..., 520:-520]
                    print(vs.shape)
                    low, median, high = (
                        np.quantile(vs, 0.05),
                        np.quantile(vs, 0.5),
                        np.quantile(vs, 0.95),
                    )

                    print("normalizing", low, median, high)
                    projected_attributions = (projected_attributions - median) / (high - low)
                    attribution_to_bigwig(
                        projected_attributions,
                        pd.DataFrame(regions),
                        dataloader["test"].dataset.chrom_size,
                        res=1,
                        mode="average",
                        output=os.path.join(
                            save_dir, f"model_{id}.attr.{wrapper}.{method}{extra}.{decay}.bigwig"
                        ),
                    )
                    np.savez(
                        os.path.join(
                            save_dir, f"model_{id}.hypo.{wrapper}.{method}{extra}.{decay}.npz"
                        ),
                        hypo,
                    )

    gc.collect()
    torch.cuda.empty_cache()
