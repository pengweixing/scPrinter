# Not organized yet, don't use~
#
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

try:
    from scprinter.utils import regionparser
except:
    from ...utils import regionparser
from .attributions import *


def multiscale_attribution(
    model,
    X,
    outputs=np.arange(99),
    group_size=1,
    n_shuffles=20,
    verbose=False,
    project=True,
    sum=True,
    method="ism",
):
    dev = model.parameters().__next__().device
    outputs = np.array(outputs)
    outputs = np.array_split(outputs, len(outputs) // group_size)
    if method != "ism":
        _references = torch.cat(
            [dinucleotide_shuffle(xx, n_shuffles=n_shuffles).to(dev) for xx in X], dim=0
        )
        _references = _references.type(X.dtype)
        attrs = []
        for output in tqdm(outputs, disable=not verbose):
            model.nth_output = torch.as_tensor(output).to(dev)
            attributions = calculate_attributions(
                model, X, method, n_shuffles, references=_references
            )
            projected_attributions = attributions
            if project:
                projected_attributions = attributions.to(dev) * X.to(dev)
            if sum:
                projected_attributions = projected_attributions.sum(dim=1)
            projected_attributions = projected_attributions.detach().cpu().numpy()
            attrs.append(projected_attributions)
        attrs = np.stack(attrs, axis=1)
        return attrs
    else:
        print("Please run with ism")


def multiscale_ism(
    model,
    printer,
    regions,
    scales=np.arange(99),
    scale_group_size=1,
    effect_range=None,
    attr_reduce_mean=True,
    mutate_positions=None,
    verbose=False,
    project=True,
    sum=True,
    batch_size=64,
):

    regions = regionparser(regions, printer, model.dna_len)
    seqs = [
        printer.genome.fetch_onehot_seq(region[0], region[1], region[2])
        for region in np.array(regions)
    ]
    attrs = [
        _multiscale_ism(
            model,
            X_0,
            scales=scales,
            scale_group_size=scale_group_size,
            effect_range=effect_range,
            attr_reduce_mean=attr_reduce_mean,
            mutate_positions=mutate_positions,
            verbose=verbose,
            project=project,
            sum=sum,
            batch_size=batch_size,
        )
        for X_0 in seqs
    ]
    return attrs


# one sequence at a time
@torch.no_grad()
def _mutation_footprints(model, X_0, batch_size=64, verbose=False):
    dev = model.parameters().__next__().device
    n_choices, seq_len = X_0.shape
    # which neuclotide in the original sequence
    X_idxs = X_0.argmax(axis=0)
    n = seq_len * (n_choices - 1)  # num of mutants
    # create a tensor of all possible perturbations
    X = torch.tile(X_0, (n, 1))
    X = X.reshape(n, n_choices, seq_len)

    # X[0,1,2] perturb the first nucleotide
    # X[3,4,5] perturb the second nucleotide...
    for k in range(1, n_choices):
        idx = np.arange(seq_len) * (n_choices - 1) + (k - 1)
        # set the original nucleotide to 0
        X[idx, X_idxs, np.arange(seq_len)] = 0
        # set the perturbed nucleotide to 1
        X[idx, (X_idxs + k) % n_choices, np.arange(seq_len)] = 1

    model = model.eval()
    reference = model(X_0[None].float().to(dev))[0]  # 0 for the shape
    starts = np.arange(0, X.shape[0], batch_size)
    ys = []
    for start in tqdm(starts, disable=not verbose):
        X_ = X[start : start + batch_size].float().to(dev)
        y = model(X_)
        y = y[0]  # get the output footprint
        ys.append(y.detach().cpu().numpy())
    ys = np.concatenate(ys, axis=0)
    ys = ys.reshape((seq_len, n_choices - 1, ys.shape[1], ys.shape[2]))
    return reference.detach().cpu().numpy(), ys


# one sequence at a time
@torch.no_grad()
def _multiscale_ism(
    model,
    X_0,
    scales=np.arange(99),
    scale_group_size=1,
    effect_range=None,
    attr_reduce_mean=True,
    mutate_positions=None,
    verbose=False,
    project=True,
    sum=True,
    batch_size=64,
):

    dev = model.parameters().__next__().device
    scales = np.array(scales)
    # partition scales into groups to speed up calculation
    scales = np.array_split(scales, len(scales) // scale_group_size)

    n_choices, seq_len = X_0.shape
    # which neuclotide in the original sequence
    X_idxs = X_0.argmax(axis=0)
    n = seq_len * (n_choices - 1)  # num of mutants
    # create a tensor of all possible perturbations
    X = torch.tile(X_0, (n, 1))
    X = X.reshape(n, n_choices, seq_len)

    # X[0,1,2] perturb the first nucleotide
    # X[3,4,5] perturb the second nucleotide...
    for k in range(1, n_choices):
        idx = np.arange(seq_len) * (n_choices - 1) + (k - 1)
        # set the original nucleotide to 0
        X[idx, X_idxs, np.arange(seq_len)] = 0
        # set the perturbed nucleotide to 1
        X[idx, (X_idxs + k) % n_choices, np.arange(seq_len)] = 1
    if mutate_positions is not None:
        X = X.reshape(seq_len, (n_choices - 1), n_choices, seq_len)
        X = X[mutate_positions]
        X = X.reshape(-1, n_choices, seq_len)
        X_idxs = X_idxs[mutate_positions]

    model = model.eval()
    reference = model(X_0[None].float().to(dev))[0]  # 0 for the shape

    def normalize(
        x,
    ):
        shapes = x.shape
        x = x.reshape((shapes[0], -1))
        x = x - x.mean(dim=-1, keepdims=True)
        x = torch.sigmoid(x) * x
        x = x.reshape(shapes)
        return x

    starts = np.arange(0, X.shape[0], batch_size)
    ism = [[] for _ in range(len(scales))]
    for start in tqdm(starts, disable=not verbose):
        X_ = X[start : start + batch_size].float().to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = model(X_)
            y = y[0]  # get the output footprint

        if effect_range is None:
            for i, scale in enumerate(scales):
                ism[i].append(
                    (normalize(y[:, scale]) - normalize(reference[:, scale])).sum(dim=(1, 2))
                )
        else:
            changed_position = torch.arange(start, start + X_.shape[0]) // (n_choices - 1)
            offset = (
                seq_len - reference.shape[-1]
            ) // 2  # (offset because, the original sequence is padded)
            # We use chrombpnet, dna_len=2114, outputshape=800 as an example
            # So when changing a sequence on position 1237, it correspond to the 1237-657=580th position in the output
            # And we should look at 580-range, 580+range in the output
            field_start = changed_position - offset - (effect_range // 2)
            field_end = changed_position - offset + (effect_range // 2)
            field_start[field_start < 0] = 0
            field_end[field_end < 0] = 0
            field_end[field_end > seq_len] = seq_len
            for i, scale in enumerate(scales):
                tmp = []
                diff = normalize(y[:, scale]) - normalize(reference[:, scale])
                for diff_, pos, s, e in zip(diff, changed_position, field_start, field_end):
                    tmp.append(diff_[..., s:e].sum())
                ism[i].append(torch.as_tensor(tmp).cuda())

    for i in range(len(scales)):
        ism[i] = torch.cat(ism[i])
    final = []
    for scale in range(len(scales)):
        isms = ism[scale].reshape(-1, n_choices - 1)  # reshaped to seq_len, n_choices - 1

        j_idxs = torch.arange(isms.shape[0])
        X_ism = torch.zeros(isms.shape[0], n_choices, device="cuda")
        for i in range(1, n_choices):
            i_idxs = (X_idxs.flatten() + i) % n_choices
            X_ism[j_idxs, i_idxs] = isms[:, i - 1].flatten()

        X_ism = X_ism.T  # reshaped to n_choices, seq_len
        X_ism = X_ism - X_ism.mean(
            dim=0, keepdims=True
        )  # X_ism calculates deltas, as in the difference between the original and the perturbed
        # minus the mean, so it represents, as compared to the other 3 nucleotides, how much the original nucleotide contributes to the prediction
        if project:
            if mutate_positions is None:
                X_ism = X_ism * X_0.to(dev)
            else:
                X_ism = X_ism * X_0[..., mutate_positions].to(dev)
        if sum:
            X_ism = X_ism.sum(dim=0)
        X_ism = X_ism.detach().cpu().numpy()
        # X_ism /= np.linalg.norm(X_ism, axis=-1, keepdims=True) # Normalize so score across scales are comparable.
        final.append(X_ism)

    return np.stack(final, axis=0)


# one sequence at a time
@torch.no_grad()
def _multiscale_ism_l2(
    model,
    X_0,
    scales=np.arange(99),
    scale_group_size=1,
    # effect_range=None,
    attr_reduce_mean=True,
    mutate_positions=None,
    specific_pos=None,
    verbose=False,
    project=True,
    sum=True,
    batch_size=64,
):

    dev = model.parameters().__next__().device
    scales = np.array(scales)
    # partition scales into groups to speed up calculation
    scales = np.array_split(scales, len(scales) // scale_group_size)
    # print (scales)
    n_choices, seq_len = X_0.shape
    # which neuclotide in the original sequence
    X_idxs = X_0.argmax(axis=0)
    n = seq_len * (n_choices - 1)  # num of mutants
    # create a tensor of all possible perturbations
    X = torch.tile(X_0, (n, 1))
    X = X.reshape(n, n_choices, seq_len)

    # X[0,1,2] perturb the first nucleotide
    # X[3,4,5] perturb the second nucleotide...
    for k in range(1, n_choices):
        idx = np.arange(seq_len) * (n_choices - 1) + (k - 1)
        # set the original nucleotide to 0
        X[idx, X_idxs, np.arange(seq_len)] = 0
        # set the perturbed nucleotide to 1
        X[idx, (X_idxs + k) % n_choices, np.arange(seq_len)] = 1
    if mutate_positions is not None:
        X = X.reshape(seq_len, (n_choices - 1), n_choices, seq_len)
        X = X[mutate_positions]
        X = X.reshape(-1, n_choices, seq_len)
        X_idxs = X_idxs[mutate_positions]

    model = model.eval()
    reference = model(X_0[None].float().to(dev))[0]  # 0 for the shape
    if specific_pos is not None:
        reference = reference[:, :, specific_pos]

    def cosine_diff(ref, mut):
        ref = ref.reshape((ref.shape[0], -1))
        mut = mut.reshape((mut.shape[0], -1))
        # 1-cosine
        # return 1 - F.cosine_similarity(ref, mut, dim=-1)
        return (ref - mut).pow(2).sum(dim=-1).sqrt() / ref.shape[-1]

        # return (ref.sum(dim=-1) - mut.sum(dim=-1)) / ref.shape[-1]

    starts = np.arange(0, X.shape[0], batch_size)
    ism = [[] for _ in range(len(scales))]
    for start in tqdm(starts, disable=not verbose):
        X_ = X[start : start + batch_size].float().to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = model(X_)
            y = y[0]  # get the output footprint
            if specific_pos is not None:
                y = y[:, :, specific_pos]

        for i, scale in enumerate(scales):
            ism[i].append(cosine_diff(reference[:, scale], y[:, scale]))

    for i in range(len(scales)):
        ism[i] = torch.cat(ism[i])
    final = []
    for scale in range(len(scales)):
        isms = ism[scale].reshape(-1, n_choices - 1)  # reshaped to seq_len, n_choices - 1

        j_idxs = torch.arange(isms.shape[0])
        X_ism = torch.zeros(isms.shape[0], n_choices, device="cuda")
        for i in range(1, n_choices):
            i_idxs = (X_idxs.flatten() + i) % n_choices
            X_ism[j_idxs, i_idxs] = isms[:, i - 1].flatten()

        X_ism = X_ism.T  # reshaped to n_choices, seq_len
        X_ism = -(
            X_ism - X_ism.mean(dim=0, keepdims=True)
        )  # X_ism calculates deltas, as in the difference between the original and the perturbed
        # minus the mean, so it represents, as compared to the other 3 nucleotides, how much the original nucleotide contributes to the prediction
        if project:
            if mutate_positions is None:
                X_ism = X_ism * X_0.to(dev)
            else:
                X_ism = X_ism * X_0[..., mutate_positions].to(dev)
        if sum:
            X_ism = X_ism.sum(dim=0)
        X_ism = X_ism.detach().cpu().numpy()
        # X_ism /= np.linalg.norm(X_ism, axis=-1, keepdims=True) # Normalize so score across scales are comparable.
        final.append(X_ism)

    return np.stack(final, axis=0)


def multiscale_visualization(attr, cropp_len=800, ax=None, rainbow=False, cbar=False):
    assert len(attr.shape) == 3, "attr should be shape (n_scales, n_choices, seq_len)"
    assert attr.shape[1] == 4, "attr should be shape (n_scales, n_choices, seq_len)"
    trim = (attr.shape[-1] - cropp_len) // 2
    if trim > 0:
        attr = attr[:, :, trim:-trim]
    vmin = 0.0
    vmax = np.quantile(attr, 0.99)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2))
    if rainbow:
        attr[attr == 0] = np.nan
        sns.heatmap(attr[0][::-1][:, 0], cmap="Greens", vmin=vmin, vmax=vmax, cbar=False)
        sns.heatmap(attr[0][::-1][:, 1], cmap="Blues", vmin=vmin, vmax=vmax, ax=ax, cbar=cbar)
        sns.heatmap(attr[0][::-1][:, 2], cmap="Oranges", vmin=vmin, vmax=vmax, ax=ax, cbar=False)
        sns.heatmap(attr[0][::-1][:, 3], cmap="Reds", vmin=vmin, vmax=vmax, ax=ax, cbar=False)
        attr = np.nan_to_num(attr)
    else:
        sns.heatmap(
            attr[0][::-1].sum(axis=1),
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar=cbar,
        )
