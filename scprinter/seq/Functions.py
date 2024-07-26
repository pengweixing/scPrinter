# Loss functions, activation functions etc.
# CumulativeCounter and CumulativePearson classes are from bolera by hqliu.
import math
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# Loss for the counts
def log1p_mse(log_predicted_counts, true_counts, reduction="mean"):
    log_true = torch.log1p(true_counts)
    return F.mse_loss(
        log_predicted_counts.reshape((-1)), log_true.reshape((-1)), reduction=reduction
    )


def batch_pearson_correlation(x, y):

    # Compute means along the batch dimension
    mean_x = torch.mean(x, dim=-1, keepdim=True)
    mean_y = torch.mean(y, dim=-1, keepdim=True)

    diff_x = x - mean_x
    diff_y = y - mean_y

    # Compute covariance and variance
    covariance = torch.sum(diff_x * diff_y, dim=-1)
    variance_x = torch.sum((diff_x) ** 2, dim=-1)
    variance_y = torch.sum((diff_y) ** 2, dim=-1)

    # Pearson correlation
    correlation = covariance / (
        torch.sqrt(variance_x * variance_y) + 1e-8
    )  # Adding small value for numerical stability
    # print (x.shape, y.shape, correlation.shape)
    return correlation


def pearson_correlation(x, y, mean_x, mean_y, bs=1e6):
    bs = int(bs)
    covariance, variance_x, variance_y = 0, 0, 0
    for i in range(0, x.shape[0], bs):
        diff_x, diff_y = (
            x[i : i + bs].to(mean_x.device) - mean_x,
            y[i : i + bs].to(mean_x.device) - mean_y,
        )
        # Compute covariance and variance
        covariance += torch.sum(diff_x * diff_y).detach().cpu().item()
        variance_x += torch.sum((diff_x) ** 2).detach().cpu().item()
        variance_y += torch.sum((diff_y) ** 2).detach().cpu().item()

    # Pearson correlation
    correlation = covariance / (
        math.sqrt(variance_x * variance_y) + 1e-8
    )  # Adding small value for numerical stability
    return correlation
