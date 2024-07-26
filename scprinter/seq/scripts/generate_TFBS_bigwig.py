import argparse

import scprinter as scp
from scprinter.seq.dataloader import *
from scprinter.seq.interpretation.attributions import *
from scprinter.seq.Models import *


def main():
    parser = argparse.ArgumentParser(description="Bindingscore BPNet")
    parser.add_argument("--count_pt", type=str, default=None, help="model.pt")
    parser.add_argument("--foot_pt", type=str, default=None, help="model.pt")
    parser.add_argument("--seq_count", type=str, nargs="+", help="seq_count", default=None)
    parser.add_argument("--seq_foot", type=str, nargs="+", help="seq_foot", default=None)
    parser.add_argument("--genome", type=str, default="hg38", help="genome")
    parser.add_argument("--peaks", type=str, default=None, help="peaks")
    parser.add_argument("--save_name", type=str, default="/", help="save_dir")
    parser.add_argument("--gpus", nargs="+", type=int, help="gpus")
    parser.add_argument("--temp", action="store_true", default=False, help="temp")
    parser.add_argument(
        "--avg", action="store_true", default=False, help="just avg without finetune model"
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="overwrite")
    parser.add_argument("--silent", action="store_true", default=False, help="silent")
    parser.add_argument("--per-peak", action="store_true", default=False, help="per-peak")

    torch.set_num_threads(4)
    args = parser.parse_args()
    seq_count = args.seq_count
    seq_foot = args.seq_foot
    count_pt = args.count_pt
    foot_pt = args.foot_pt
    genome = args.genome
    peaks = args.peaks
    gpus = args.gpus
    save_name = args.save_name
    overwrite = args.overwrite
    silent = args.silent
    torch.cuda.set_device(gpus[0])
    if save_name[-1] != "_":
        save_name += "_"
    if count_pt is not None:
        count_model = torch.jit.load(count_pt, map_location="cpu")
        count_model.with_motif = False
        count_model = count_model.to("cuda")
        count_model.eval()
    else:
        count_model = None
    if foot_pt is not None:
        foot_model = torch.jit.load(foot_pt, map_location="cpu")
        foot_model.with_motif = False
        foot_model = foot_model.to("cuda")
        foot_model.eval()
    else:
        foot_model = None

    if count_model is None and foot_model is None:
        raise ValueError("No model loaded")
    elif count_model is None:
        count_model = foot_model
        seq_count = seq_foot
    elif foot_model is None:
        foot_model = count_model
        seq_foot = seq_count

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
    summits = summits[summits[0].isin(split["test"])]
    signal_window = 1200 // 2
    genome = args.genome
    if genome == "hg38":
        genome = scp.genome.hg38
    elif genome == "mm10":
        genome = scp.genome.mm10
    else:
        raise ValueError("genome not supported")

    seq_count = [pyBigWig.open(i, "r") for i in seq_count] if seq_count is not None else []
    seq_foot = [pyBigWig.open(i, "r") for i in seq_foot] if seq_foot is not None else []

    count_feats = []
    foot_feats = []
    batch_size = 512

    count_preds = []
    foot_preds = []
    for i in trange(len(summits), disable=silent):
        chrom, summit = summits.iloc[i]

        count_feat = np.nanmean(
            [
                np.nan_to_num(
                    i.values(chrom, summit - signal_window, summit + signal_window, numpy=True)
                )
                for i in seq_count
            ],
            axis=0,
        )
        foot_feat = np.nanmean(
            [
                np.nan_to_num(
                    i.values(chrom, summit - signal_window, summit + signal_window, numpy=True)
                )
                for i in seq_foot
            ],
            axis=0,
        )

        count_feats.append(count_feat)
        foot_feats.append(foot_feat)

        if len(count_feats) == batch_size and (not args.avg):
            count_feats = np.array(count_feats)
            foot_feats = np.array(foot_feats)

            if args.per_peak:
                low, median, high = (
                    np.quantile(count_feats, 0.05, axis=1, keepdims=True),
                    np.quantile(count_feats, 0.5, axis=1, keepdims=True),
                    np.quantile(count_feats, 0.95, axis=1, keepdims=True),
                )
                print(count_feats.shape)
                count_feats = (count_feat - median) / (high - low)

                low, median, high = (
                    np.quantile(foot_feats, 0.05, axis=1, keepdims=True),
                    np.quantile(foot_feats, 0.5, axis=1, keepdims=True),
                    np.quantile(foot_feats, 0.95, axis=1, keepdims=True),
                )
                foot_feats = (foot_feat - median) / (high - low)

            count_feats = torch.as_tensor(count_feats).float().cuda()[:, None, :]
            foot_feats = torch.as_tensor(foot_feats).float().cuda()[:, None, :]
            count_feats_rev = torch.flip(count_feats, [2])
            foot_feats_rev = torch.flip(foot_feats, [2])
            count_pred = torch.stack(
                [
                    torch.sigmoid(count_model(count_feats)).cpu().detach()[:, 0, :],
                    torch.flip(
                        torch.sigmoid(count_model(count_feats_rev)).cpu().detach()[:, 0, :],
                        [1],
                    ),
                ],
                axis=0,
            )
            foot_pred = torch.stack(
                [
                    torch.sigmoid(foot_model(foot_feats)).cpu().detach()[:, 0, :],
                    torch.flip(
                        torch.sigmoid(foot_model(foot_feats_rev)).cpu().detach()[:, 0, :],
                        [1],
                    ),
                ],
                axis=0,
            )
            count_pred = torch.mean(count_pred, 0)
            foot_pred = torch.mean(foot_pred, 0)
            # count_pred,_ = torch.max(count_pred, 0)
            # foot_pred,_ = torch.max(foot_pred, 0)

            count_preds.append(count_pred)
            foot_preds.append(foot_pred)
            count_feats = []
            foot_feats = []

    if len(count_feats) > 0 and (not args.avg):
        count_feats = np.array(count_feats)
        foot_feats = np.array(foot_feats)
        count_feats = torch.as_tensor(count_feats).float().cuda()[:, None, :]
        foot_feats = torch.as_tensor(foot_feats).float().cuda()[:, None, :]
        count_feats_rev = torch.flip(count_feats, [2])
        foot_feats_rev = torch.flip(foot_feats, [2])
        count_pred = torch.stack(
            [
                torch.sigmoid(count_model(count_feats)).cpu().detach()[:, 0, :],
                torch.flip(
                    torch.sigmoid(count_model(count_feats_rev)).cpu().detach()[:, 0, :], [1]
                ),
            ],
            axis=0,
        )
        foot_pred = torch.stack(
            [
                torch.sigmoid(foot_model(foot_feats)).cpu().detach()[:, 0, :],
                torch.flip(torch.sigmoid(foot_model(foot_feats_rev)).cpu().detach()[:, 0, :], [1]),
            ],
            axis=0,
        )
        count_pred = torch.mean(count_pred, 0)
        foot_pred = torch.mean(foot_pred, 0)
        # count_pred, _ = torch.max(count_pred, 0)
        # foot_pred, _ = torch.max(foot_pred, 0)
        count_preds.append(count_pred)
        foot_preds.append(foot_pred)
    if args.avg:
        count_pred = np.stack(count_feats, 0)[..., 100:-100]
        seq_pred = np.stack(foot_feats, 0)[..., 100:-100]
    else:
        count_pred = torch.cat(count_preds, 0).numpy()
        seq_pred = torch.cat(foot_preds, 0).numpy()
        # np.savez(save_name + "TFBS.npz",
        #          count_pred = count_pred,
        #          seq_pred = seq_pred)

    avg = (count_pred + seq_pred) / 2
    regions = summits.copy()
    regions[1] = regions["summits"] - signal_window + 100
    regions[2] = regions["summits"] + signal_window - 100
    regions = regions[[0, 1, 2]]
    if not silent:
        print(regions, avg.shape)

    attribution_to_bigwig(
        avg,
        regions,
        genome.chrom_sizes,
        res=1,
        mode="average",
        output=(save_name + "TFBS.bigwig") if not args.avg else (save_name + "avg.bigwig"),
        verbose=not silent,
    )
    if args.temp:
        attribution_to_bigwig(
            count_pred,
            regions,
            genome.chrom_sizes,
            res=1,
            mode="average",
            output=(
                (save_name + "TFBS_count.bigwig")
                if not args.avg
                else (save_name + "avg_count.bigwig")
            ),
            verbose=not silent,
        )
        attribution_to_bigwig(
            seq_pred,
            regions,
            genome.chrom_sizes,
            res=1,
            mode="average",
            output=(
                (save_name + "TFBS_foot.bigwig")
                if not args.avg
                else (save_name + "avg_foot.bigwig")
            ),
            verbose=not silent,
        )


if __name__ == "__main__":
    main()
