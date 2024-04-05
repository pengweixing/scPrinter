import numpy as np
import torch
from tqdm.auto import tqdm, trange


@torch.no_grad()
def ism(model, X_0, args=None, batch_size=128, verbose=False):
    """In-silico mutagenesis saliency scores.

    This function will perform in-silico mutagenesis in a naive manner, i.e.,
    where each input sequence has a single mutation in it and the entirety
    of the sequence is run through the given model. It returns the ISM score,
    which is a vector of the L2 difference between the reference sequence
    and the perturbed sequences with one value for each output of the model.

    Parameters
    ----------
    model: torch.nn.Module
        The model to use.

    X_0: torch.tensor, shape=(batch_size, 4, seq_len)
        The one-hot encoded sequence to calculate saliency for.

    args: tuple or None, optional
        Additional arguments to pass into the forward function. If None,
        pass nothing additional in. Default is None.

    batch_size: int, optional
        The size of the batches.

    verbose: bool, optional
        Whether to display a progress bar as positions are being mutated. One
        display bar will be printed for each sequence being analyzed. Default
        is False.

    Returns
    -------
    X_ism: torch.tensor, shape=(batch_size, 4, seq_len)
        The saliency score for each perturbation.
    """
    # print (X_0.shape)
    n_seqs, n_choices, seq_len = X_0.shape
    print(X_0.shape)
    X_idxs = X_0.argmax(axis=1)

    n = seq_len * (n_choices - 1)
    X = torch.tile(X_0, (n, 1, 1))
    X = X.reshape(n, n_seqs, n_choices, seq_len).permute(1, 0, 2, 3)

    for i in range(n_seqs):
        for k in range(1, n_choices):
            idx = np.arange(seq_len) * (n_choices - 1) + (k - 1)

            X[i, idx, X_idxs[i], np.arange(seq_len)] = 0
            X[i, idx, (X_idxs[i] + k) % n_choices, np.arange(seq_len)] = 1

    model = model.eval()

    if args is None:
        reference = model(X_0).unsqueeze(1)
    else:
        reference = model(X_0, *args).unsqueeze(1)

    starts = np.arange(0, X.shape[1], batch_size)
    isms = []
    for i in range(n_seqs):
        ism = []
        for start in tqdm(starts, disable=not verbose):
            X_ = X[i, start : start + batch_size].cuda()

            if args is None:
                y = model(X_)
            else:
                args_ = tuple(a[i : i + 1] for a in args)
                y = model(X_, *args_)

            ism.append(y - reference[i])

        ism = torch.cat(ism)
        if len(ism.shape) > 1:
            ism = ism.sum(dim=list(range(1, len(ism.shape))))
        isms.append(ism)

    isms = torch.stack(isms)
    isms = isms.reshape(n_seqs, seq_len, n_choices - 1)

    j_idxs = torch.arange(n_seqs * seq_len)
    X_ism = torch.zeros(n_seqs * seq_len, n_choices, device="cuda")
    for i in range(1, n_choices):
        i_idxs = (X_idxs.flatten() + i) % n_choices
        X_ism[j_idxs, i_idxs] = isms[:, :, i - 1].flatten()

    X_ism = X_ism.reshape(n_seqs, seq_len, n_choices).permute(0, 2, 1)
    X_ism = X_ism - X_ism.mean(dim=1, keepdims=True)
    return X_ism
