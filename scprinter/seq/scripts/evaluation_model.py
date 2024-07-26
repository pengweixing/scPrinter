import argparse
import gc
import subprocess
from concurrent.futures import ProcessPoolExecutor

import h5py

import scprinter as scp
from scprinter.seq.dataloader import *
from scprinter.seq.interpretation.attribution_wrapper import *
from scprinter.seq.interpretation.attributions import *
from scprinter.seq.Models import *

parser = argparse.ArgumentParser(description="Bindingscore BPNet")
parser.add_argument("--pt", type=str, default="config.JSON", help="model.pt")
parser.add_argument("--models", type=str, default=None, help="model")
parser.add_argument("--genome", type=str, default="hg38", help="genome")
parser.add_argument("--peaks", type=str, default=None, help="peaks")
parser.add_argument("--start", type=int, default=0, help="start index of peaks")
parser.add_argument("--end", type=int, default=-1, help="end index of peaks")
parser.add_argument("--method", nargs="+", type=str, default=None, help="method")
parser.add_argument("--wrapper", nargs="+", type=str, default=None, help="method")
parser.add_argument("--nth_output", nargs="+", type=str, default=None, help="nth output")
parser.add_argument("--write_numpy", action="store_true", default=False, help="write numpy")
parser.add_argument("--gpus", nargs="+", type=int, help="gpus")
parser.add_argument("--overwrite", action="store_true", default=False, help="overwrite")
parser.add_argument("--decay", type=float, default=None, help="decay")
parser.add_argument("--extra", type=str, default="", help="extra")
parser.add_argument("--model_norm", type=str, default=None, help="key for model norm")
parser.add_argument("--sample", type=int, default=None, help="sample")
parser.add_argument("--silent", action="store_true", default=False, help="silent")


def main():

    torch.set_num_threads(4)
    args = parser.parse_args()
    verbose = not args.silent
    write_bigwig = not args.write_numpy
    if args.models is not None:
        ids = args.models
        ids = ids.split(",")
        ids = [int(i) for i in ids]
    else:
        ids = [None]

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

    peaks = args.peaks
    summits = pd.read_table(peaks, sep="\t", header=None)
    summits = summits.drop_duplicates([0, 1, 2])  # drop exact same loci
    summits["summits"] = (summits[1] + summits[2]) // 2
    summits = summits[[0, "summits"]]
    summits["summits"] = np.array(summits["summits"], dtype=int)

    acc_model = torch.load(args.pt, map_location="cpu").cuda()

    # If there's coverage, set it to be the same
    if acc_model.coverages is not None:
        mm = acc_model.coverages.weight.data[:, -1].mean()
        acc_model.coverages.weight.data[:, -1] = mm

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

    if ids[0] is not None:
        dataset = seq2PRINTDataset(
            signals=signals,
            ref_seq=genome.fetch_fa(),
            summits=summits,
            DNA_window=dna_len,
            signal_window=signal_window,
            max_jitter=0,
            min_counts=None,
            max_counts=None,
            cached=False,
            reverse_compliment=False,
            device="cpu",
        )

        dataloader = seq2PRINTDataLoader(
            dataset=dataset, batch_size=64, num_workers=10, pin_memory=True, shuffle=False
        )

        summits = dataset.summits
        sample = args.sample

        #  This is the lora mode, parallel in terms of models:
        regions = np.array(
            [summits[:, 0], summits[:, 1] - dna_len // 2, summits[:, 1] + dna_len // 2]
        ).T
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
                    vs_collection = []
                    extra = f"_{n_out}_"
                    if len(gpus) > 1:
                        # dataset['test'].cache()
                        n_split = len(gpus)

                        unfinished_ids = []
                        for id in ids:
                            if os.path.exists(
                                os.path.join(
                                    save_dir,
                                    f"model_{id}.hypo.{wrapper}.{method}{extra}.{args.decay}.npz",
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
                                "seq2print_attr",
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
                        bar = tqdm(ids)
                        dataset.cache()
                        for id in bar:
                            bar.set_description(f"working on {id}")
                            if os.path.exists(
                                os.path.join(
                                    save_dir,
                                    f"model_{id}.hypo.{wrapper}.{method}{extra}.{args.decay}.npz",
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

                            if wrapper == "just_sum":
                                model = JustSumWrapper(model_0, nth_output=n_out, threshold=0.301)
                            elif wrapper == "count":
                                model = CountWrapper(model_0)
                            model = model.cuda()

                            sample = args.sample
                            if sample is not None:
                                random_ids = torch.randperm(len(dataset.cache_seqs))[:sample]
                            else:
                                random_ids = slice(None)
                            attributions = calculate_attributions(
                                model,
                                X=dataset.cache_seqs[random_ids],
                                n_shuffles=20,
                                method=method,
                                verbose=verbose,
                            )
                            projected_attributions = project_attrs(
                                attributions, dataset.cache_seqs[random_ids], 64, "cuda"
                            )
                            hypo, ohe = (
                                attributions.detach().cpu().numpy(),
                                dataset.cache_seqs[random_ids].detach().cpu().numpy(),
                            )
                            # vs = projected_attributions
                            # low, median, high = np.quantile(vs, 0.05), np.quantile(vs, 0.5), np.quantile(vs, 0.95)
                            vs = projected_attributions[..., 520:-520]

                            if args.sample is not None:
                                vs_collection.append(vs)
                                continue
                            norm_key = args.model_norm
                            if norm_key is not None:
                                if norm_key == "count":
                                    low, median, high = acc_model.count_norm
                                elif norm_key == "footprint":
                                    low, median, high = acc_model.foot_norm
                            else:
                                low, median, high = (
                                    np.quantile(vs, 0.05),
                                    np.quantile(vs, 0.5),
                                    np.quantile(vs, 0.95),
                                )
                            if verbose:
                                print("normalizing", low, median, high)

                            projected_attributions = (projected_attributions - median) / (
                                high - low
                            )
                            attribution_to_bigwig(
                                projected_attributions,
                                pd.DataFrame(regions),
                                dataloader.dataset.chrom_size,
                                res=1,
                                mode="average",
                                output=os.path.join(
                                    save_dir,
                                    f"model_{id}.attr.{wrapper}.{method}{extra}.{args.decay}.bigwig",
                                ),
                            )
                            np.savez(
                                os.path.join(
                                    save_dir,
                                    f"model_{id}.hypo.{wrapper}.{method}{extra}.{args.decay}.npz",
                                ),
                                hypo,
                            )

                        if args.sample is not None:
                            vs_collection = np.concatenate(vs_collection, axis=0)
                            print("sampled signals for normalization", vs_collection.shape)
                            vs = vs_collection.reshape((-1))
                            # Only trying to figure out the normalization factor
                            low, median, high = (
                                np.quantile(vs, 0.05),
                                np.quantile(vs, 0.5),
                                np.quantile(vs, 0.95),
                            )
                            print("normalizing", low, median, high)
                            np.save(
                                os.path.join(
                                    save_dir, f"norm.{wrapper}.{method}{extra}.{args.decay}.npy"
                                ),
                                np.array([low, median, high]),
                            )
                            gc.collect()
                            torch.cuda.empty_cache()
                            continue

            gc.collect()
            torch.cuda.empty_cache()
    else:
        sample = args.sample
        if sample is not None:
            summits = summits.sample(sample)
        dataset = seq2PRINTDataset(
            signals=signals,
            ref_seq=genome.fetch_fa(),
            summits=summits,
            DNA_window=dna_len,
            signal_window=signal_window,
            max_jitter=0,
            min_counts=None,
            max_counts=None,
            cached=False,
            reverse_compliment=False,
            device="cpu",
        )

        dataloader = seq2PRINTDataLoader(
            dataset=dataset, batch_size=64, num_workers=10, pin_memory=True, shuffle=False
        )

        summits = dataset.summits

        #  This is the normal mode, parallel in terms of peaks:
        start = args.start
        end = args.end
        if end == -1:
            end = summits.shape[0]

        summits = summits[start:end]
        regions = np.array(
            [summits[:, 0], summits[:, 1] - dna_len // 2, summits[:, 1] + dna_len // 2]
        ).T
        acc_model.upsample = False
        if len(gpus) > 1:
            print(acc_model)

        # nth_output = torch.as_tensor([5, 10]) - 1
        params_combination = []
        save_dir = f"{args.pt}_deepshap"
        while not os.path.exists(save_dir):
            try:
                print("making dir")
                os.makedirs(save_dir)
            except:
                pass

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

                        n_split = len(gpus)
                        bs = int(math.ceil(len(summits) / n_split))

                        starts = [i * bs for i in range(n_split)]
                        ends = [(i + 1) * bs for i in range(n_split)]

                        commands = [
                            [
                                "seq2print_attr",
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
                            n_out = torch.as_tensor(
                                [i for i in range(int(n_out_start), int(n_out_end))]
                            )
                        else:
                            n_out = torch.as_tensor([int(n_out)])
                        if n_out[0] < 0:
                            n_out = None

                        if wrapper == "just_sum":
                            model = JustSumWrapper(acc_model, nth_output=n_out, threshold=0.301)
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
                        dataset.cache()
                        attributions = calculate_attributions(
                            model,
                            X=dataset.cache_seqs[start:end],
                            n_shuffles=20,
                            method=method,
                            verbose=True,
                        )
                        projected_attributions = project_attrs(
                            attributions, dataset.cache_seqs[start:end], 64, "cuda"
                        )
                        hypo, ohe = (
                            attributions.detach().cpu().numpy(),
                            dataset.cache_seqs[start:end].detach().cpu().numpy(),
                        )

                        # projected_attributions = np.zeros((end - start, 2114))
                    if write_bigwig:
                        print(projected_attributions.shape, hypo.shape, ohe.shape, regions)
                        np.savez(
                            os.path.join(
                                save_dir, f"hypo.{wrapper}.{method}{extra}.{args.decay}.npz"
                            ),
                            hypo,
                        )
                        # np.savez(os.path.join(save_dir, f'ohe.{wrapper}.{method}{extra}.{args.decay}.npz'), ohe)
                        vs = projected_attributions[..., 520:-520]
                        print(vs.shape)
                        if args.sample is not None:
                            # Only trying to figure out the normalization factor
                            low, median, high = (
                                np.quantile(vs, 0.05),
                                np.quantile(vs, 0.5),
                                np.quantile(vs, 0.95),
                            )
                            print("normalizing", low, median, high)
                            np.save(
                                os.path.join(
                                    save_dir, f"norm.{wrapper}.{method}{extra}.{args.decay}.npy"
                                ),
                                np.array([low, median, high]),
                            )
                            gc.collect()
                            torch.cuda.empty_cache()
                            continue

                        norm_key = args.model_norm
                        if norm_key is not None:
                            if norm_key == "count":
                                low, median, high = acc_model.count_norm
                            elif norm_key == "footprint":
                                low, median, high = acc_model.foot_norm

                        else:
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
                            dataloader.dataset.chrom_size,
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
                                os.path.join(
                                    save_dir, f"ohe.{wrapper}.{method}{extra}.{start}-{end}.npz"
                                ),
                                ohe,
                            )

            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
