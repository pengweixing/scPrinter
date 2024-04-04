import argparse
import gc
import json

import pandas as pd
import torch

from bpnetfootprint.attributions import *
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
    # "valid": [
    #     "chr8",
    #     "chr20"
    # ],
    # "train": [
    #     "chr2",
    #     "chr4",
    #     "chr5",
    #     "chr7",
    #     "chr9",
    #     "chr10",
    #     "chr11",
    #     "chr12",
    #     "chr13",
    #     "chr14",
    #     "chr15",
    #     "chr16",
    #     "chr17",
    #     "chr18",
    #     "chr19",
    #     "chr21",
    #     "chr22",
    #     "chrX",
    #     "chrY"
    # ]
}


replicate = config["replicate"]
for rep in range(1):
    summits = pd.read_table("/data/rzhang/PRINT_rev/K562/peaks.bed", sep="\t", header=None)
    summits = summits.drop_duplicates([0, 1, 2])  # drop exact same loci
    summits["summits"] = summits[1] - 1 + 500
    summits = summits[[0, "summits"]]

    # summits = pd.DataFrame({0:np.array(['chr6']), 'summits':np.array([154733378])})
    # summits = pd.concat([summits]*10, axis=0)
    print(summits)
    savename = config["savename"] + f"_{rep}"
    datasets = {
        k: ChromBPDataset(
            signals=["/data/rzhang/PRINT_rev/K562/bindingscore.bw"],
            ref_seq="/home/rzhang/Data/hg38/hg38.fa",
            summits=summits[summits[0].isin(split[k])],
            DNA_window=2114,
            signal_window=1000,
            max_jitter=0,
            min_counts=None,
            max_counts=None,
            cached=True,
            reverse_compliment=False,
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

    acc_model = torch.load(savename + ".model.pt", map_location="cpu").cuda()
    acc_model.eval()
    y_all = []
    pred_all = []
    for data in tqdm(dataloader["test"]):
        X, y = data
        pred_score = acc_model.predict(X, batch_size=X.shape[0])
        y_all.append(y)
        pred_all.append(pred_score)
    y_all = torch.cat(y_all)
    pred_all = torch.cat(pred_all)
    print(y_all.shape, pred_all.shape)

    summits = dataloader["test"].dataset.summits
    width = pred_all.shape[-1]
    regions = np.array([summits[:, 0], summits[:, 1] - width // 2, summits[:, 1] + width // 2]).T
    attribution_to_bigwig(
        (
            pred_all[:, 0, :].cpu().numpy() / 10 + 0.5
            if config["loss"] == "regression"
            else torch.sigmoid(pred_all[:, 0, :].cpu()).numpy()
        ),
        pd.DataFrame(regions),
        dataloader["test"].dataset.chrom_size,
        res=1,
        mode="average",
        output=savename + ".bigwig",
    )
    if "attr_method" in config:
        methods = config["attr_method"]
    else:
        methods = ["deeplift", "shap", "inputxgradient"]  # 'integrated_gradients',
    #
    # methods = ['deeplift','deeplift_hypo', 'shap','shap_hypo','inputxgradient_norm','integrated_gradients', 'ism'] #
    summits = summits
    regions = np.array([summits[:, 0], summits[:, 1] - 1057, summits[:, 1] + 1057]).T
    acc_model.upsample = False
    for method in methods:
        print("method", method)
        attributions = calculate_attributions(
            ProfileWrapperFootprintClass(acc_model),
            model_output=None,
            X=datasets["test"].cache_seqs,
            n_shuffles=20,
            method=method,
            verbose=True,
        )
        projected_attributions = projected_shap(
            attributions, datasets["test"].cache_seqs, 64, "cuda"
        )
        attribution_to_bigwig(
            projected_attributions,
            pd.DataFrame(regions),
            dataloader["test"].dataset.chrom_size,
            res=1,
            mode="average",
            output=savename + ".attr." + method + ".bigwig",
        )

    del acc_model
    gc.collect()
    torch.cuda.empty_cache()
