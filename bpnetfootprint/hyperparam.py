import numpy as np
import torch


def bias_model_hyperparam(
    train_peak_dataset,
    valid_peak_dataset,
    train_nonpeak_dataset,
    valid_nonpeak_dataset,
    bias_threshold_factor=0.5,
    lower_threshold=0.0001,
    upper_threshold=0.9999,
):

    train_cov = train_peak_dataset.coverage.sum(axis=-1)
    valid_cov = valid_peak_dataset.coverage.sum(axis=-1)
    train_valid_cov = np.concatenate([train_cov, valid_cov], axis=0)

    counts_threshold = np.quantile(train_valid_cov, 0.01) * bias_threshold_factor
    print("counts_threshold", counts_threshold)

    train_nonpeak_dataset.filter_by_coverage(-1, counts_threshold)
    valid_nonpeak_dataset.filter_by_coverage(-1, counts_threshold)

    train_nonpeak_cov = train_nonpeak_dataset.coverage.sum(axis=-1)
    valid_nonpeak_cov = valid_nonpeak_dataset.coverage.sum(axis=-1)
    train_valid_nonpeak_cov = np.concatenate([train_nonpeak_cov, valid_nonpeak_cov], axis=0)

    upper_threshold = np.quantile(train_valid_nonpeak_cov, upper_threshold)
    lower_threshold = np.quantile(train_valid_nonpeak_cov, lower_threshold)
    print("upper_threshold", upper_threshold)
    print("lower_threshold", lower_threshold)
    train_nonpeak_dataset.filter_by_coverage(lower_threshold, upper_threshold)
    valid_nonpeak_dataset.filter_by_coverage(lower_threshold, upper_threshold)
    print(
        "Number of nonpeaks after applying upper-bound cut-off and removing outliers : ",
        len(train_nonpeak_dataset) + len(valid_nonpeak_dataset),
    )

    train_nonpeak_cov = train_nonpeak_dataset.coverage.sum(axis=-1)
    valid_nonpeak_cov = valid_nonpeak_dataset.coverage.sum(axis=-1)
    train_valid_nonpeak_cov = np.concatenate([train_nonpeak_cov, valid_nonpeak_cov], axis=0)

    count_loss_weight = np.median(train_valid_nonpeak_cov) / 10
    print("count_loss_weight", count_loss_weight)

    if count_loss_weight < 1.0:
        count_loss_weight = 1.0
        print("WARNING: you are training on low-read depth data")

    return counts_threshold, lower_threshold, upper_threshold, count_loss_weight


def chrombpnet_model_hyperparam(
    train_peak_dataset,
    valid_peak_dataset,
    train_nonpeak_dataset,
    valid_nonpeak_dataset,
    negative_sampling_ratio=0.1,
    lower_threshold=0.0001,
    upper_threshold=0.9999,
):
    train_cov = train_peak_dataset.coverage.sum(axis=-1)
    valid_cov = valid_peak_dataset.coverage.sum(axis=-1)
    train_valid_cov = np.concatenate([train_cov, valid_cov], axis=0)

    train_nonpeak_cov = train_nonpeak_dataset.coverage.sum(axis=-1)
    valid_nonpeak_cov = valid_nonpeak_dataset.coverage.sum(axis=-1)
    train_valid_nonpeak_cov = np.concatenate([train_nonpeak_cov, valid_nonpeak_cov], axis=0)
    neg_index = np.random.permutation(len(train_valid_nonpeak_cov))[
        : int(len(train_valid_cov) * negative_sampling_ratio)
    ]
    final_counts = np.concatenate([train_valid_cov, train_valid_nonpeak_cov[neg_index]], axis=0)

    upper_threshold = np.quantile(final_counts, upper_threshold)
    lower_threshold = np.quantile(final_counts, lower_threshold)

    print("upper_threshold", upper_threshold)
    print("lower_threshold", lower_threshold)
    train_peak_dataset.filter_by_coverage(lower_threshold, upper_threshold)
    valid_peak_dataset.filter_by_coverage(lower_threshold, upper_threshold)
    train_nonpeak_dataset.filter_by_coverage(lower_threshold, upper_threshold)
    valid_nonpeak_dataset.filter_by_coverage(lower_threshold, upper_threshold)

    print(
        "Number of peaks after removing outliers: ",
        len(train_peak_dataset) + len(valid_peak_dataset),
    )
    print(
        "Number of nonpeaks after removing outliers: ",
        len(train_nonpeak_dataset) + len(valid_nonpeak_dataset),
    )

    # somehow it uses <= and >= istead of < and >, consistent with their code
    count_loss_weight = (
        np.median(
            final_counts[(final_counts >= lower_threshold) & (final_counts <= upper_threshold)]
        )
        / 10
    )
    print("count_loss_weight", count_loss_weight)
    return lower_threshold, upper_threshold, count_loss_weight
