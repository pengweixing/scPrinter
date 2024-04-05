import gc
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import transformers
import wandb

import scprinter as scp
from scprinter.seq.attribution_wrapper import *
from scprinter.seq.attributions import *
from scprinter.seq.dataloader import *
from scprinter.seq.ema import EMA
from scprinter.seq.minimum_footprint import *
from scprinter.seq.Models import *
from scprinter.seq.Modules import *
from scprinter.utils import load_entire_hdf5, loadDispModel


def construct_model_from_config(config):
    n_filters = config["n_filters"]
    n_layers = config["n_layers"]
    activation = config["activation"]
    head_kernel_size = config["head_kernel_size"]
    kernel_size = config["kernel_size"]
    if activation == "relu":
        activation = torch.nn.ReLU()
    elif activation == "gelu":
        activation = torch.nn.GELU()
    batch_norm = config["batch_norm"]
    dilation_base = config["dilation_base"]
    bottleneck_factor = config["bottleneck_factor"]
    bottleneck = int(n_filters * bottleneck_factor)
    rezero = config["rezero"]
    batch_norm_momentum = config["batch_norm_momentum"]
    groups = config["groups"]
    no_inception = config["no_inception"]
    n_inception_layers = config["n_inception_layers"]
    inception_layers_after = config["inception_layers_after"]
    if no_inception:
        n_inception_layers = 0

    inception_version = config["inception_version"]
    if inception_layers_after:
        inception_bool = [False] * (n_layers - n_inception_layers) + [True] * (n_inception_layers)
    else:
        inception_bool = [True] * n_inception_layers + [False] * (n_layers - n_inception_layers)

    acc_dna_cnn = DNA_CNN(
        n_filters=n_filters,
    )
    dilation_func = lambda x: 2 ** (x + dilation_base)
    acc_hidden = DilatedCNN(
        n_filters=n_filters,
        bottleneck=bottleneck,
        n_layers=n_layers,
        kernel_size=kernel_size,
        groups=groups,
        activation=activation,
        batch_norm=batch_norm,
        residual=True,
        rezero=rezero,
        dilation_func=dilation_func,
        batch_norm_momentum=batch_norm_momentum,
        inception=inception_bool,
        inception_version=inception_version,
    )

    acc_head = Footprints_head(
        n_filters, kernel_size=head_kernel_size, n_scales=99, per_peak_feats=1
    )
    output_len = 800
    dna_len = output_len + acc_dna_cnn.conv.weight.shape[2] - 1
    for i in range(n_layers):
        dna_len = dna_len + 2 * (kernel_size // 2) * dilation_func(i)
    print("dna_len", dna_len)

    acc_model = scFootprintBPNet(
        dna_cnn_model=acc_dna_cnn,
        hidden_layer_model=acc_hidden,
        profile_cnn_model=acc_head,
        dna_len=dna_len,
        output_len=output_len,
    )
    return acc_model, dna_len, output_len


def main(
    config=None,
    enable_wandb=True,
):
    # Initialize a new wandb run
    with wandb.init(config=config):
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        disp_path = scp.datasets.pretrained_dispersion_model
        dispmodel = loadDispModel(disp_path)
        dispmodel = dispModel(deepcopy(dispmodel)).cuda()

        print("start a new run!!!")
        print("run id", wandb.run.name)
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        data_dir = config["data_dir"]
        temp_dir = config["temp_dir"]
        model_dir = config["model_dir"]

        split = config["split"]
        signals = os.path.join(data_dir, config["signals"])
        peaks = os.path.join(data_dir, config["peaks"])
        genome = config["genome"]
        if genome == "hg38":
            genome = scp.genome.hg38
        elif genome == "mm10":
            genome = scp.genome.mm10
        else:
            raise ValueError("genome not supported")
        bias = str(genome.fetch_bias())[:-3] + ".bw"
        print("run name", wandb.run.name)

        signals = [signals, bias]
        peaks = pd.read_table(peaks, sep="\t", header=None)
        summits = peaks
        summits["summits"] = (summits[1] + summits[2]) // 2
        summits = summits[[0, "summits"]]
        summits["index"] = np.arange(len(summits))
        print(summits)
        modes = np.arange(2, 101, 1)

        max_jitter = config["max_jitter"]

        ema_flag = config["ema"]
        amp_flag = config["amp"]
        batch_size = config["batch_size"]
        weight_decay = config["weight_decay"]

        acc_model, dna_len, output_len = construct_model_from_config(config)
        acc_model = acc_model.cuda()

        print("output len", output_len)
        print("model")
        print(acc_model)
        total_params = sum(p.numel() for p in acc_model.parameters() if p.requires_grad)
        print("total params", total_params)
        acc_model.dna_len = dna_len
        acc_model.signal_len = output_len

        datasets = {
            k: ChromBPDataset(
                signals=signals,
                ref_seq=genome.fetch_fa(),
                summits=summits[summits[0].isin(split[k])],
                DNA_window=dna_len,
                signal_window=output_len + 200,
                max_jitter=max_jitter if k in ["train"] else 0,
                min_counts=None,
                max_counts=None,
                cached=True,
                lazy_cache=True,
                reverse_compliment=(config["reverse_compliment"] if k in ["train"] else False),
                device="cpu",
            )
            for k in split
        }

        coverage = datasets["train"].coverage

        min_, max_ = np.quantile(coverage, 0.0001), np.quantile(coverage, 0.9999)
        min_ = max(min_, 10)
        print("coverage cutoff", min_, max_)
        for k in split:
            datasets[k].filter_by_coverage(min_, max_)

        dataloader = {
            k: ChromBPDataLoader(
                dataset=datasets[k],
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=True,
            )
            for k in split
        }

        ema = None
        if ema_flag:
            update_after_step = 100
            print("update after step", update_after_step)
            ema = EMA(
                acc_model,
                beta=0.9999,  # exponential moving average factor
                update_after_step=update_after_step,  # only after this number of .update() calls will it start updating
                update_every=10,
            )  # how often to actually update, to save on compute (updates every 10th .update() call)

        torch.cuda.empty_cache()
        optimizer = torch.optim.AdamW(
            acc_model.parameters(), lr=config["lr"], weight_decay=weight_decay
        )
        if "scheduler" in config:
            scheduler = config["scheduler"]
        else:
            scheduler = False
        if scheduler:
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=3000, num_training_steps=100000
            )
        else:
            scheduler = None
        acc_model.fit(
            dispmodel,
            dataloader["train"],
            validation_data=dataloader["valid"],
            validation_size=None,
            max_epochs=300,
            optimizer=optimizer,
            scheduler=scheduler,
            validation_freq=None,
            early_stopping=(5 if "early_stopping" not in config else config["early_stopping"]),
            return_best=True,
            savename=f"{temp_dir}/{wandb.run.name}",
            modes=modes,
            downsample=None if "downsample" not in config else config["downsample"],
            ema=ema,
            use_amp=amp_flag,
            use_wandb=enable_wandb,
        )
        if ema:
            del acc_model
            acc_model = torch.load(f"{temp_dir}/{wandb.run.name}.ema_model.pt").cuda()
        acc_model.eval()
        savename = config["savename"]
        torch.save(acc_model, f"{model_dir}/{savename}-{wandb.run.name}.pt")
        valid_loss, valid_within, valid_across = validation_step_footprint(
            acc_model, dataloader["valid"], None, dispmodel, modes
        )
        valid_loss, valid_within, valid_across, valid_cov = (
            float(valid_loss[0]),
            float(valid_within[0]),
            float(valid_across[0]),
            float(valid_across[1]),
        )
        test_loss, test_within, test_across = validation_step_footprint(
            acc_model, dataloader["test"], None, dispmodel, modes
        )
        test_loss, test_within, test_across, test_cov = (
            float(test_loss[0]),
            float(test_within[0]),
            float(test_across[0]),
            float(test_across[1]),
        )
        if enable_wandb:
            wandb.summary["final_valid_loss"] = valid_loss
            wandb.summary["final_valid_within"] = valid_within
            wandb.summary["final_valid_across"] = valid_across
            wandb.summary["final_test_loss"] = test_loss
            wandb.summary["final_test_within"] = test_within
            wandb.summary["final_test_across"] = test_across
            wandb.summary["final_valid_cov"] = valid_cov
            wandb.summary["final_test_cov"] = test_cov

        if "shap" in config:
            shap = config["shap"]
        else:
            shap = False
        if shap:
            summits = summits[summits[0].isin(split["test"])]
            if len(summits) > 30000:
                print("downsample summits")
                summits = summits.sample(30000)
            print(summits)
            dataset = ChromBPDataset(
                signals=signals,
                ref_seq=genome.fetch_fa(),
                summits=summits,
                DNA_window=dna_len,
                signal_window=output_len + 200,
                max_jitter=0,
                min_counts=None,
                max_counts=None,
                cached=True,
                lazy_cache=True,
                reverse_compliment=False,
                device="cpu",
            )

            acc_model.eval()
            model = ProfileWrapperFootprintClass(
                acc_model, nth_output=list(np.arange(30)), res=1, reduce_mean=True
            )
            model = model.cuda()

            attributions = calculate_attributions(
                model,
                X=dataset.cache_seqs,
                n_shuffles=20,
                method="shap_hypo",
                verbose=True,
            ).numpy()

            ohe = dataset.cache_seqs.numpy()
            print(ohe.shape, attributions.shape)
            np.savez(
                f"{model_dir}/{savename}-{wandb.run.name}-attributions.npz",
                attributions,
            )
            np.savez(f"{model_dir}/{savename}-{wandb.run.name}-ohe.npz", ohe)
        wandb.finish()

        del acc_model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
