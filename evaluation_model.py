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
parser.add_argument("--genome", type=str, default="hg38", help="genome")
parser.add_argument("--peaks", type=str, default=None, help="peaks")
parser.add_argument("--start", type=int, default=0, help="start index of peaks")
parser.add_argument("--end", type=int, default=-1, help="end index of peaks")
parser.add_argument("--method", nargs="+", type=str, default=None, help="method")
parser.add_argument("--wrapper", nargs="+", type=str, default=None, help="method")
parser.add_argument("--nth_output", nargs="+", type=str, default=None, help="nth output")
parser.add_argument("--write_numpy", action="store_true", default=False, help="write numpy")
parser.add_argument("--save_pred", action="store_true", default=False, help="save pred")
parser.add_argument("--gpus", nargs="+", type=int, help="gpus")
parser.add_argument("--overwrite", action="store_true", default=False, help="overwrite")
parser.add_argument("--replicate", type=int, default=None, help="replicate")
parser.add_argument("--decay", type=float, default=None, help="decay")
parser.add_argument("--disable_ema", action="store_true", default=False, help="disable emas")
parser.add_argument("--extra", type=str, default="", help="extra")


torch.set_num_threads(4)
args = parser.parse_args()
start = args.start
end = args.end
write_bigwig = not args.write_numpy
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
summits = summits[[0, "summits"]]
summits["summits"] = np.array(summits["summits"], dtype=int)


acc_model = torch.load(args.pt, map_location="cpu").cuda()
acc_model.eval()
dna_len = acc_model.dna_len

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
        num_workers=10,
        pin_memory=True,
        shuffle=True if k in ["train"] else False,
    )
    for k in split
}

summits = dataloader["test"].dataset.summits

start = args.start
end = args.end
if end == -1:
    end = summits.shape[0]

summits = summits[start:end]
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
            extra = args.extra + f"_{n_out}_"
            if len(gpus) > 1:
                if (
                    write_bigwig
                    and os.path.exists(
                        os.path.join(
                            save_dir,
                            f"attr.{wrapper}.{method}{extra}.{args.decay}.bigwig",
                        )
                    )
                    and (not args.overwrite)
                ):
                    print("exists")
                    continue

                # datasets['test'].cache()
                n_split = len(gpus)
                bs = int(math.ceil(len(summits) / n_split))

                starts = [i * bs for i in range(n_split)]
                ends = [(i + 1) * bs for i in range(n_split)]

                commands = [
                    [
                        "python",
                        "evaluation_model.py",
                        "--pt",
                        args.pt,
                        "--genome",
                        args.genome,
                        "--peaks",
                        args.peaks,
                        "--start",
                        str(start),
                        "--end",
                        str(end),
                        "--method",
                        method,
                        "--write_numpy",
                        "--wrapper",
                        wrapper,
                        "--nth_output",
                        str(n_out),
                        "--gpus",
                        str(gpu),
                    ]
                    + (["--overwrite"] if args.overwrite else [])
                    for i, (start, end, gpu) in enumerate(zip(starts, ends, gpus))
                ]
                print(commands)
                pool = ProcessPoolExecutor(max_workers=len(gpus))
                for gpu, command in zip(gpus, commands):
                    pool.submit(subprocess.run, command)

                pool.shutdown(wait=True)

                attributions = []
                hypo, ohe = [], []
                for start, end in zip(starts, ends):
                    attributions.append(
                        np.load(
                            os.path.join(
                                save_dir,
                                f"attr.{wrapper}.{method}{extra}.{start}-{end}.npy",
                            )
                        )
                    )
                    hypo.append(
                        np.load(
                            os.path.join(
                                save_dir,
                                f"hypo.{wrapper}.{method}{extra}.{start}-{end}.npz",
                            )
                        )["arr_0"]
                    )
                    ohe.append(
                        np.load(
                            os.path.join(
                                save_dir,
                                f"ohe.{wrapper}.{method}{extra}.{start}-{end}.npz",
                            )
                        )["arr_0"]
                    )

                projected_attributions = np.concatenate(attributions, axis=0)
                hypo = np.concatenate(hypo, axis=0)
                ohe = np.concatenate(ohe, axis=0)

            else:
                if "," in n_out:
                    n_out = n_out.split(",")
                    n_out = torch.as_tensor([int(i) for i in n_out])
                elif "-" in n_out:
                    n_out_start, n_out_end = n_out.split("-")
                    n_out = torch.as_tensor([i for i in range(int(n_out_start), int(n_out_end))])
                else:
                    n_out = torch.as_tensor([int(n_out)])
                if n_out[0] < 0:
                    n_out = None
                if wrapper == "classification":
                    model = ProfileWrapperFootprintClass(
                        acc_model,
                        nth_output=n_out,
                        res=1,
                        reduce_mean=False,
                        decay=args.decay,
                    )
                elif wrapper == "regression":
                    model = ProfileWrapperFootprint(
                        acc_model, nth_output=n_out, res=1, reduce_mean=False
                    )
                elif wrapper == "classification_reduce":
                    model = ProfileWrapperFootprintClass(
                        acc_model,
                        nth_output=n_out,
                        res=1,
                        reduce_mean=True,
                        decay=args.decay,
                    )
                elif wrapper == "regression_reduce":
                    model = ProfileWrapperFootprint(
                        acc_model, nth_output=n_out, res=1, reduce_mean=True
                    )
                elif wrapper == "just_sum":
                    model = JustSumWrapper(acc_model, nth_output=n_out, res=1, threshold=0.301)
                elif wrapper == "count":
                    model = CountWrapper(acc_model)
                model = model.cuda()
                print("method", method, wrapper)
                if (
                    write_bigwig
                    and os.path.exists(
                        os.path.join(
                            save_dir,
                            f"attr.{wrapper}.{method}{extra}.{args.decay}.bigwig",
                        )
                    )
                    and (not args.overwrite)
                ):
                    print("exists")
                    continue
                elif (
                    not write_bigwig
                    and os.path.exists(
                        os.path.join(
                            save_dir,
                            f"attr.{wrapper}.{method}{extra}.{start}-{end}.npy",
                        )
                    )
                    and (not args.overwrite)
                ):
                    print("exists")
                    continue
                datasets["test"].cache()
                attributions = calculate_attributions(
                    model,
                    X=datasets["test"].cache_seqs[start:end],
                    n_shuffles=20,
                    method=method,
                    verbose=True,
                )
                projected_attributions = projected_shap(
                    attributions, datasets["test"].cache_seqs[start:end], 64, "cuda"
                )
                hypo, ohe = (
                    attributions.detach().cpu().numpy(),
                    datasets["test"].cache_seqs[start:end].detach().cpu().numpy(),
                )

                # projected_attributions = np.zeros((end - start, 2114))
            if write_bigwig:
                print(projected_attributions.shape, hypo.shape, ohe.shape, regions)
                np.savez(
                    os.path.join(save_dir, f"hypo.{wrapper}.{method}{extra}.{args.decay}.npz"),
                    hypo,
                )
                # np.savez(os.path.join(save_dir, f'ohe.{wrapper}.{method}{extra}.{args.decay}.npz'), ohe)
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
                        save_dir, f"attr.{wrapper}.{method}{extra}.{args.decay}.bigwig"
                    ),
                )

                if len(gpus) > 1:
                    for start, end in zip(starts, ends):
                        os.remove(
                            os.path.join(
                                save_dir,
                                f"attr.{wrapper}.{method}{extra}.{start}-{end}.npy",
                            )
                        )
                        os.remove(
                            os.path.join(
                                save_dir,
                                f"hypo.{wrapper}.{method}{extra}.{start}-{end}.npz",
                            )
                        )
                        os.remove(
                            os.path.join(
                                save_dir,
                                f"ohe.{wrapper}.{method}{extra}.{start}-{end}.npz",
                            )
                        )
            else:
                if len(gpus) <= 1:
                    np.save(
                        os.path.join(
                            save_dir,
                            f"attr.{wrapper}.{method}{extra}.{start}-{end}.npy",
                        ),
                        projected_attributions,
                    )
                    np.savez(
                        os.path.join(
                            save_dir,
                            f"hypo.{wrapper}.{method}{extra}.{start}-{end}.npz",
                        ),
                        hypo,
                    )
                    np.savez(
                        os.path.join(save_dir, f"ohe.{wrapper}.{method}{extra}.{start}-{end}.npz"),
                        ohe,
                    )

    gc.collect()
    torch.cuda.empty_cache()
