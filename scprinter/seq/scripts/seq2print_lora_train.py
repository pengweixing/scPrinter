import argparse
import gc
import json
import os.path
import pickle
import shutil
import socket
from pathlib import Path

import pandas as pd
import transformers
import wandb
from ema_pytorch import EMA

import scprinter as scp
from scprinter.seq.dataloader import *
from scprinter.seq.Models import *
from scprinter.utils import loadDispModel

torch.backends.cudnn.benchmark = True


def entry(config=None, wandb_run_name="", enable_wandb=True):
    # Initialize a new wandb run
    print("start a new run!!!")
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    disp_path = scp.datasets.pretrained_dispersion_model
    dispmodel = loadDispModel(disp_path)
    dispmodel = dispModel(deepcopy(dispmodel)).cuda()

    config = wandb.config

    data_dir = config["data_dir"]
    temp_dir = config["temp_dir"]
    model_dir = config["model_dir"]

    split = config["split"]
    peaks = os.path.join(data_dir, config["peaks"])

    insertion = os.path.join(data_dir, config["insertion"])
    grp2barcodes = os.path.join(data_dir, config["grp2barcodes"])
    grp2embeddings = os.path.join(data_dir, config["grp2embeddings"])
    pretrain_model = os.path.join(data_dir, config["pretrain_model"])
    # coverages = os.path.join(data_dir, config["coverages"])

    peaks = pd.read_table(peaks, sep="\t", header=None)
    summits = peaks
    summits["summits"] = (summits[1] + summits[2]) // 2

    summits = summits[[0, "summits"]]
    summits["index"] = np.arange(len(summits))
    print(summits)
    modes = np.arange(2, 101, 1)
    insertion = pickle.load(open(insertion, "rb"))
    grp2barcodes = np.array(pickle.load(open(grp2barcodes, "rb")), dtype="object")
    grp2embeddings = np.array(pickle.load(open(grp2embeddings, "rb")))
    print(grp2embeddings)

    max_jitter = config["max_jitter"]
    ema_flag = config["ema"]
    amp_flag = config["amp"]
    batch_size = config["batch_size"]
    weight_decay = config["weight_decay"]
    genome = config["genome"]
    cell_sample = config["cell_sample"]
    if genome == "hg38":
        genome = scp.genome.hg38
    elif genome == "mm10":
        genome = scp.genome.mm10
    else:
        raise ValueError("genome not supported")

    lr = config["lr"]
    acc_model = torch.load(pretrain_model)
    acc_model.cuda()
    for p in acc_model.parameters():
        p.requires_grad = False
    # acc_model.profile_cnn_model.conv_layer.weight.requires_grad = True
    # acc_model.profile_cnn_model.conv_layer.bias.requires_grad = True
    # acc_model.profile_cnn_model.linear.weight.requires_grad = True
    # acc_model.profile_cnn_model.linear.bias.requires_grad = True
    # acc_model, dna_len, output_len = construct_model_from_config(config)
    acc_model.cuda()

    total_params = sum(p.numel() for p in acc_model.parameters())
    print("total params - pretrained model", total_params)

    print("output len", acc_model.output_len)
    print("dna len", acc_model.dna_len)

    # if os.path.exists(coverages):
    #     cov = pickle.load(open(coverages, "rb"))
    #     save_coverage = False
    # else:
    #     cov = {k: None for k in split}
    #     save_coverage = True
    save_coverage = False
    cov = {k: None for k in split}
    datasets = {
        k: scseq2PRINTDataset(
            insertion_dict=insertion,
            bias=str(genome.fetch_bias())[:-3] + ".bw",
            group2idx=grp2barcodes,
            ref_seq=genome.fetch_fa(),
            summits=summits[summits[0].isin(split[k])],
            DNA_window=acc_model.dna_len,
            signal_window=acc_model.output_len + 200,
            max_jitter=max_jitter if k in ["train"] else 0,
            cached=True,
            lazy_cache=True,
            reverse_compliment=(config["reverse_compliment"] if k in ["train"] else False),
            device="cpu",
            coverages=cov[k],
            data_augmentation=True if k in ["train"] else False,
            mode=config["dataloader_mode"] if k in ["train"] else "uniform",
            cell_sample=cell_sample,
        )
        for k in split
    }

    # if save_coverage:
    #     cov = {k: datasets[k].coverages for k in split}
    #     pickle.dump(cov, open(coverages, "wb"))

    dataloader = {
        k: seq2PRINTDataLoader(
            dataset=datasets[k],
            batch_size=(
                (batch_size // cell_sample) * 2
                if (config["dataloader_mode"] == "peak") and (k in ["train"])
                else batch_size
            ),
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            collate_fn=(
                collate_fn_singlecell
                if (config["dataloader_mode"] == "peak") and (k in ["train"])
                else None
            ),
        )
        for k in split
    }

    val_loss, profile_pearson, across_pearson_fp, across_pearson_cov = validation_step_footprint(
        acc_model,
        dataloader["valid"],
        (len(datasets["valid"].summits) // batch_size),
        dispmodel,
        modes,
    )
    print("before lora", val_loss, profile_pearson, across_pearson_fp, across_pearson_cov)
    # optimizer = torch.optim.AdamW(acc_model.parameters(), lr=1e-3, weight_decay=weight_decay)
    # if "scheduler" in config:
    #     scheduler = config["scheduler"]
    # else:
    #     scheduler = False
    # if scheduler:
    #     scheduler = transformers.get_cosine_schedule_with_warmup(
    #         optimizer, num_warmup_steps=3000, num_training_steps=100000
    #     )
    # else:
    #     scheduler = None
    #
    # acc_model.cuda()
    # ema = None
    # if ema_flag:
    #     update_after_step = 100
    #     print("update after step", update_after_step)
    #     ema = EMA(
    #         acc_model,
    #         beta=0.9999,  # exponential moving average factor
    #         update_after_step=update_after_step,  # only after this number of .update() calls will it start updating
    #         update_every=10,
    #     )  # how often to actually update, to save on compute (updates every 10th .update() call)
    #
    # acc_model.fit(
    #     dispmodel,
    #     dataloader["train"],
    #     validation_data=dataloader["valid"],
    #     validation_size=(len(datasets["valid"].summits) // batch_size),
    #     max_epochs=3,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     validation_freq=(len(datasets["train"].summits) // batch_size),
    #     early_stopping=(3 if "early_stopping" not in config else config["early_stopping"]),
    #     return_best=True,
    #     savename=f"{temp_dir}/{wandb_run_name}",
    #     modes=modes,
    #     downsample=None if "downsample" not in config else config["downsample"],
    #     ema=ema,
    #     use_amp=amp_flag,
    #     use_wandb=enable_wandb,
    #     batch_size=batch_size,
    # )
    #
    # if ema:
    #     del acc_model
    #     acc_model = torch.load(f"{temp_dir}/{wandb_run_name}.ema_model.pt").cuda()
    #
    # val_loss, profile_pearson, across_pearson_fp, across_pearson_cov = validation_step_footprint(
    #     acc_model,
    #     dataloader["valid"],
    #     (len(datasets["valid"].summits) // batch_size),
    #     dispmodel,
    #     modes,
    # )
    #
    # print("after adjusted output head", val_loss, profile_pearson, across_pearson_fp, across_pearson_cov)
    #
    # for p in acc_model.parameters():
    #     p.requires_grad = False

    acc_model = acc_model.cpu()
    acc_model = seq2PRINT(
        dna_cnn_model=acc_model.dna_cnn_model,
        hidden_layer_model=acc_model.hidden_layer_model,
        profile_cnn_model=acc_model.profile_cnn_model,
        dna_len=acc_model.dna_len,
        output_len=acc_model.output_len,
        embeddings=grp2embeddings,
        rank=config["lora_rank"],
        hidden_dim=config["lora_hidden_dim"],
        lora_dna_cnn=config["lora_dna_cnn"],
        lora_dilated_cnn=config["lora_dilated_cnn"],
        lora_pff_cnn=config["lora_pff_cnn"],
        lora_profile_cnn=config["lora_profile_cnn"],
        lora_count_cnn=config["lora_count_cnn"],
        n_lora_layers=config["n_lora_layers"],
    )
    acc_model = acc_model.cuda()

    print("model")
    print(acc_model)
    total_params = sum(p.numel() for p in acc_model.parameters() if p.requires_grad)
    print("total trainable params", total_params)
    total_params = sum(p.numel() for p in acc_model.parameters())
    print("total params", total_params)

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
    optimizer = torch.optim.AdamW(acc_model.parameters(), lr=lr, weight_decay=weight_decay)
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

    val_loss, profile_pearson, across_pearson_fp, across_pearson_cov = validation_step_footprint(
        acc_model,
        dataloader["valid"],
        (len(datasets["valid"].summits) // batch_size),
        dispmodel,
        modes,
    )
    print("before lora", val_loss, profile_pearson, across_pearson_fp, across_pearson_cov)
    # raise EOFError
    acc_model.fit(
        dispmodel,
        dataloader["train"],
        validation_data=dataloader["valid"],
        validation_size=(len(datasets["valid"].summits) // batch_size),
        max_epochs=300,
        optimizer=optimizer,
        scheduler=scheduler,
        validation_freq=(len(datasets["train"].summits) // batch_size),
        early_stopping=(5 if "early_stopping" not in config else config["early_stopping"]),
        return_best=True,
        savename=f"{temp_dir}/{wandb_run_name}",
        modes=modes,
        downsample=None if "downsample" not in config else config["downsample"],
        ema=ema,
        use_amp=amp_flag,
        use_wandb=enable_wandb,
        accumulate_grad=config["accumulate_grad_batches"],
        batch_size=batch_size,
        coverage_warming=1,
    )
    if ema:
        del acc_model
        acc_model = torch.load(f"{temp_dir}/{wandb_run_name}.ema_model.pt").cuda()
    acc_model.eval()
    savename = config["savename"]
    torch.save(acc_model, f"{model_dir}/{savename}-{wandb_run_name}.pt")
    val_loss, profile_pearson, across_pearson_fp, across_pearson_cov = validation_step_footprint(
        acc_model,
        dataloader["valid"],
        (len(datasets["valid"].summits) // batch_size),
        dispmodel,
        modes,
    )

    test_loss, test_profile_pearson, test_across_pearson_fp, test_across_pearson_cov = (
        validation_step_footprint(
            acc_model,
            dataloader["test"],
            (len(datasets["valid"].summits) // batch_size),
            dispmodel,
            modes,
        )
    )
    # Now that the model is trained we use it to estimate the normalization range of the shap values
    # For LoRA model the tricky part is that we also need to sample across multiple cells.
    model_type = "lora"
    model = f"{model_dir}/{savename}-{wandb_run_name}.pt"
    gpus = [0]
    peak_file = os.path.join(data_dir, config["peaks"])
    num_cell = len(grp2barcodes)
    count_hypo = scp.tl.seq_attr_seq2print(
        peak_file,
        model_type,
        model,
        genome,
        gpus,
        preset="count",
        sample=max(50000 // num_cell, 2),
        overwrite=True,
        launch=True,
        lora_ids=list(np.arange(num_cell)),
    )
    foot_hypo = scp.tl.seq_attr_seq2print(
        peak_file,
        model_type,
        model,
        genome,
        gpus,
        preset="footprint",
        sample=max(50000 // num_cell, 2),
        overwrite=True,
        launch=True,
        lora_ids=list(np.arange(num_cell)),
    )
    count_hypo = np.load(count_hypo)
    foot_hypo = np.load(foot_hypo)
    low, median, high = count_hypo
    print("count head normalization factor", low, median, high)
    acc_model.count_norm = [low, median, high]
    low, median, high = foot_hypo
    print("footprint head normalization factor", low, median, high)
    acc_model.foot_norm = [low, median, high]
    torch.save(acc_model, f"{model_dir}/{savename}-{wandb_run_name}.pt")
    shutil.rmtree(f"{model_dir}/{savename}-{wandb_run_name}.pt_deepshap")

    if enable_wandb:
        wandb.summary["final_valid_loss"] = val_loss
        wandb.summary["final_valid_within"] = profile_pearson
        wandb.summary["final_valid_across"] = across_pearson_fp
        wandb.summary["final_valid_cov"] = across_pearson_cov
        wandb.summary["final_test_loss"] = test_loss
        wandb.summary["final_test_within"] = test_profile_pearson
        wandb.summary["final_test_across"] = test_across_pearson_fp
        wandb.summary["final_test_cov"] = test_across_pearson_cov
        wandb.finish()
        del acc_model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="seq2PRINT LoRA model")
    parser.add_argument("--config", type=str, default="config.JSON", help="config file")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/",
        help="data directory, will be append to all data path in config",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="/",
        help="temp directory, will be used to store working models",
    )
    parser.add_argument(
        "--model_dir", type=str, default="/", help="will be used to store final models"
    )
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb")
    parser.add_argument("--project", type=str, default="scPrinterSeq_v3_lora", help="project name")
    torch.set_num_threads(4)
    args = parser.parse_args()
    config = json.load(open(args.config))

    for path in [args.data_dir, args.temp_dir, args.model_dir]:
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    config["data_dir"] = args.data_dir
    config["temp_dir"] = args.temp_dir
    config["model_dir"] = args.model_dir

    if args.enable_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.project,
            notes=socket.gethostname() if "notes" not in config else config["notes"],
            # track hyperparameters and run metadata
            tags=config["tags"] if "tags" in config else [],
            config=config,
            job_type="training",
            reinit=True,
        )

    # Initialize a new wandb run
    if args.enable_wandb:
        with wandb.init(config=config):
            config = wandb.config
            print("run id", wandb.run.name)
            print("run name", wandb.run.name)
            wandb_run_name = wandb.run.name
            entry(config, wandb_run_name, args.enable_wandb)
    else:
        config = config
        wandb_run_name = ""
        entry(config, wandb_run_name, args.enable_wandb)


if __name__ == "__main__":
    main()
