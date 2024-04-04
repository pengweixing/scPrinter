import argparse
import gc
import json

import pandas as pd
import torch

from bpnetfootprint.chromBPNet_Modules import *
from bpnetfootprint.dataloader import *
from bpnetfootprint.Enformer_modules import *
from bpnetfootprint.evaluation import *
from bpnetfootprint.Modules import *

parser = argparse.ArgumentParser(description="Bindingscore BPNet")
parser.add_argument("--config", type=str, default="config.JSON", help="config file")
torch.set_num_threads(4)
args = parser.parse_args()
config = json.load(open(args.config))

split = {
    "test": ["chr1", "chr3", "chr6"],
    "valid": ["chr8", "chr20"],
    "train": [
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
    ],
}

replicate = config["replicate"]
for rep in range(replicate):
    savename = config["savename"] + f"_{rep}"

    if config["model"] == "chrombpnet":
        n_filters = config["n_filters"]
        n_layers = config["n_layers"]
        activation = config["activation"]
        head_kernel_size = config["head_kernel_size"]
        if activation == "relu":
            activation = torch.nn.ReLU()
        elif activation == "gelu":
            activation = torch.nn.GELU()
        batch_norm = config["batch_norm"]

        acc_dna_cnn = acc_dna_cnn = DNA_CNN(
            n_filters=n_filters,
        )

        acc_dilated_cnn = DilatedCNN(
            n_filters=n_filters,
            n_layers=n_layers,
            kernel_size=3,
            activation=activation,
            batch_norm=batch_norm,
            residual=True,
        )
        acc_head = Bindingscore_head(
            n_filters,
            kernel_size=[head_kernel_size],
            pool_size=10,
            upsample_size=10,
            pool_mode="avg",
        )
        acc_model = BindingScoreBPNet(
            dna_cnn_model=acc_dna_cnn,
            hidden_layer_model=acc_dilated_cnn,
            profile_cnn_model=acc_head,
            output_len=1000,
        )
        acc_model = acc_model.cuda()
        output_len = 1000
        dna_len = 2114
    elif config["model"] == "enformer":
        from einops.layers.torch import Rearrange

        n_dilated_layers = config["n_dilated_layers"]
        n_conv_layers = config["n_conv_layers"]
        n_trans_layers = config["n_trans_layers"]
        n_filters = config["n_filters"]
        activation = config["activation"]
        if activation == "relu":
            activation = torch.nn.ReLU()
        elif activation == "gelu":
            activation = torch.nn.GELU()
        batch_norm = config["batch_norm"]
        pool_size = config["pool_size"]
        if "simple_attention" not in config:
            simple_attention = False
        else:
            simple_attention = config["simple_attention"]

        acc_dna_cnn = DNA_CNN_Enformer(
            n_filters=n_filters,
            kernel_size=21,
            padding=10,
            in_channels=4,
            pool_size=pool_size,
            batch_norm=batch_norm,
        ).cuda()

        acc_hidden = []
        if n_dilated_layers > 0:
            acc_hidden.append(
                DilatedCNN(
                    n_filters=n_filters,
                    n_layers=n_dilated_layers,
                    kernel_size=3,
                    activation=activation,
                    batch_norm=batch_norm,
                    residual=True,
                )
            )
        if n_conv_layers > 0:
            acc_hidden.append(
                ConvTower(
                    in_channels=n_filters,
                    out_channels=768 if n_trans_layers else n_filters,
                    kernel_size=5,
                    pool_size=pool_size,
                    batch_norm=batch_norm,
                    n_layers=n_conv_layers,
                )
            )
        if n_trans_layers > 0:
            acc_hidden.append(Rearrange("b d n -> b n d"))
            acc_hidden.append(
                AttentionTower(
                    input_dim=768,
                    n_head=8,
                    d_model=768,
                    d_k=64,
                    attn_dropout=0.05,
                    pos_dropout=0.01,
                    dropout=0.4,
                    n_layers=n_trans_layers,
                    simple_attention=simple_attention,
                )
            )
            acc_hidden.append(Rearrange("b n d -> b d n"))

        acc_hidden = nn.Sequential(*acc_hidden).cuda()
        # print(acc_hidden)
        upsample_size = 10 if pool_size == 1 else pool_size ** (n_conv_layers + 1)
        final_pool_size = 10 if pool_size == 1 else 1
        acc_head = Bindingscore_head(
            768 if n_trans_layers else n_filters,
            kernel_size=[1],
            pool_size=final_pool_size,
            upsample_size=upsample_size,
            pool_mode="avg",
        ).cuda()
        dna_len = 2114 if pool_size == 1 else int(2114 // upsample_size * upsample_size)

        output_len_needed_in_X = int(1000 / upsample_size * final_pool_size)
        x_shape = acc_hidden(acc_dna_cnn(torch.zeros(1, 4, dna_len).float().cuda())).shape[-1]
        trim = (x_shape - output_len_needed_in_X) // 2
        x_shape_final = x_shape - trim * 2
        output_len_final = int(x_shape_final / final_pool_size * upsample_size)
        print(
            dna_len,
            output_len_needed_in_X,
            x_shape,
            trim,
            x_shape_final,
            output_len_final,
        )
        output_len = output_len_final

        acc_model = BindingScoreBPNet(
            dna_cnn_model=acc_dna_cnn,
            hidden_layer_model=acc_hidden,
            profile_cnn_model=acc_head,
            output_len=output_len,
        )

        acc_model = acc_model.cuda()
    else:
        pass

    print("output len", output_len)
    print("model")
    print(acc_model)

    summits = pd.read_table("/data/rzhang/PRINT_rev/K562/peaks.bed", sep="\t", header=None)
    summits = summits.drop_duplicates([0, 1, 2])  # drop exact same loci
    summits["summits"] = summits[1] - 1 + 500
    summits = summits[[0, "summits"]]
    datasets = {
        k: ChromBPDataset(
            signals=["/data/rzhang/PRINT_rev/K562/bindingscore.bw"],
            ref_seq="/home/rzhang/Data/hg38/hg38.fa",
            summits=summits[summits[0].isin(split[k])],
            DNA_window=dna_len,
            signal_window=output_len,
            max_jitter=config["max_jitter"] if k in ["train"] else 0,
            min_counts=None,
            max_counts=None,
            cached=True,
            reverse_compliment=(config["reverse_compliment"] if k in ["train"] else False),
            device="cpu",
        )
        for k in split
    }
    dataloader = {
        k: ChromBPDataLoader(
            dataset=datasets[k],
            batch_size=config["batch_size"],
            num_workers=0,
            pin_memory=True,
            shuffle=True if k in ["train"] else False,
        )
        for k in split
    }

    if config["loss"] == "regression":
        for k in datasets:
            dataset = datasets[k]
            dataset.cache_signals -= 0.5
            dataset.cache_signals *= 10

    torch.cuda.empty_cache()
    weight_decay = config["weight_decay"]
    optimizer = torch.optim.AdamW(acc_model.parameters(), lr=1e-3, weight_decay=weight_decay)
    n_epoch = acc_model.fit(
        dataloader["train"],
        validation_data=dataloader["valid"],
        validation_size=None,
        mode=[config["loss"]],
        max_epochs=100,
        optimizer=optimizer,
        scheduler=None,
        validation_freq=None,
        early_stopping=5,
        return_best=True,
        savename=savename,
    )
    torch.save(acc_model, savename + ".model.pt")
    valid_loss, valid_within, valid_across = validation_step_footprint(
        acc_model, dataloader["valid"], None, mode=[config["loss"]], verbose=True
    )
    valid_loss, valid_within, valid_across = (
        float(valid_loss[0]),
        float(valid_within[0]),
        float(valid_across[0]),
    )
    test_loss, test_within, test_across = validation_step_footprint(
        acc_model, dataloader["test"], None, mode=[config["loss"]], verbose=True
    )
    test_loss, test_within, test_across = (
        float(test_loss[0]),
        float(test_within[0]),
        float(test_across[0]),
    )

    config["n_epoch"] = n_epoch
    config["valid_loss"] = valid_loss
    config["valid_within"] = valid_within
    config["valid_across"] = valid_across
    config["test_loss"] = test_loss
    config["test_within"] = test_within
    config["test_across"] = test_across
    json.dump(config, open(savename + ".log", "w"))
    del acc_model
    gc.collect()
    torch.cuda.empty_cache()
    # break
