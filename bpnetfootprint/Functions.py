# Loss functions, activation functions etc.
# BPNet losses are from https://github.com/jmschrei/bpnet-lite/blob/master/bpnetlite/losses.pyå
import math

import torch
import torch.nn.functional as F
from torch import nn


def multinomial_negative_log_likelihood(logps, true_counts):
    log_fact_sum = torch.lgamma(torch.sum(true_counts, dim=-1) + 1)
    log_prod_fact = torch.sum(torch.lgamma(true_counts + 1), dim=-1)
    log_prod_exp = torch.sum(true_counts * logps, dim=-1)
    return -log_fact_sum + log_prod_fact - log_prod_exp


def log1p_mse(log_predicted_counts, true_counts, reduction="mean"):
    log_true = torch.log1p(true_counts)
    return F.mse_loss(
        log_predicted_counts.reshape((-1)), log_true.reshape((-1)), reduction=reduction
    )


def shape_loss(pred, true, reduction="mean"):
    true_proba = true / torch.sum(true, dim=-1, keepdim=True)
    # print (true_proba.shape, pred.shape, true_proba)
    return F.cross_entropy(pred, true_proba, reduction=reduction)
    # return multinomial_negative_log_likelihood(pred, true).mean()


# relative positional encoding functions
# Positional encoding from enformer
def get_positional_features_exponential(positions, features, seq_len, min_half_life=3.0):
    max_range = math.log(seq_len) / math.log(2.0)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device=positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.0) / half_life * positions)


def get_positional_features_central_mask(positions, features, seq_len):
    center_widths = 2 ** torch.arange(1, features + 1, device=positions.device).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1.0, x) - rate * x
    log_normalization = torch.lgamma(concentration) - concentration * torch.log(rate)
    return torch.exp(log_unnormalized_prob - log_normalization)


def get_positional_features_gamma(
    positions, features, seq_len, stddev=None, start_mean=None, eps=1e-8
):
    if stddev is None:
        stddev = seq_len / (2 * features)

    if start_mean is None:
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device=positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev**2
    probabilities = gamma_pdf(positions.float().abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim=-1, keepdim=True)
    return outputs


def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma,
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(
            f"feature size is not divisible by number of components ({num_components})"
        )

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1)
    return embeddings


def relative_shift(x):
    # x shape of (batch, heads, len_q, len_k)
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    # x shape of (batch, heads, len_q, len_k + 1)
    _, h, q, k = x.shape
    x = x.reshape(-1, h, k, q)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, q, k - 1)
    return x[..., : ((k + 1) // 2)]
