import torch


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

    return correlation
