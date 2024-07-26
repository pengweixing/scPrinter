import torch
import torch.nn as nn

try:
    from scprinter.utils import zscore2pval_torch
except ImportError:
    from ...utils import zscore2pval_torch


class FootprintScaling(torch.nn.Module):
    """
    A torch module that transform the z-score to pvalues to get the footprint scores.
    This class is necessary for captum or deepshap to work
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return zscore2pval_torch(x)


class Smoother(nn.Module):
    """
    A torch module that smooths the predicted footprints across the scale axis (useful to get seqattr that emphasize on a specifc scale but still incorporating other scales a little bit)

    Parameters
    ----------
    mode: int
        The mode of the smoothing kernel.
    """

    def __init__(self, mode, decay):
        super().__init__()
        self.mode = mode
        self.decay = decay

        self.layer1 = torch.nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(99, 1), bias=False
        )
        # initialize:
        with torch.no_grad():
            v = decay ** (torch.abs(torch.arange(99) - mode))
            v = v / v.sum()
            self.layer1.weight.data[0, 0, :, 0] = v
        self.layer1.weight.requires_grad = False

    def forward(self, x):
        x = self.layer1(x)
        return x


class JustSumWrapper(torch.nn.Module):
    """
    A wrapper class that returns the sum of the predicted footprints

    Parameters
    ----------
    model: torch.nn.Module
        A trained seq2PRINT torch model to be wrapped.
    nth_output: int | torch.Tensor | None
        Which scale(s) to be used. If None, all scales (2-99bp) will be used.
    specific_pos: int | slice | torch.Tensor | None
        Which genomic position(s) to be used. If None, all positions will be used. This is useful when calculating footprints at specific positions.
    weight: torch.Tensor | None

    """

    def __init__(
        self,
        model: torch.nn.Module,
        nth_output=None,
        specific_pos=None,
        weight=None,
        threshold=None,
    ):
        super().__init__()
        self.model = model
        self.nth_output = nth_output
        self.specific_pos = specific_pos
        self.weight = weight
        self.threshold = threshold
        self.relu = torch.nn.ReLU()
        self.scaling = FootprintScaling()

    def forward(self, X, *args, **kwargs):
        logits = self.model(X, *args, **kwargs)[0]
        if self.nth_output is not None:
            logits = logits[:, self.nth_output]
        if self.specific_pos is not None:
            logits = logits[:, :, self.specific_pos]
        logits = logits.reshape(X.shape[0], -1)
        # Such that global average moving won't affect the result
        logits = logits - torch.mean(logits, dim=-1, keepdims=True)
        logits = self.scaling(logits)
        if self.threshold is not None:
            logits = self.relu(logits - self.threshold)
        if self.weight is not None:
            logits = logits * self.weight

        # if self.reduce_mean:
        #     logits = logits - torch.mean(logits, dim=-1, keepdims=True)
        # y = self.scaling(logits)
        y = logits
        return (y).sum(axis=-1, keepdims=True)


FootprintWrapper = JustSumWrapper


class CountWrapper(torch.nn.Module):
    """A wrapper class that only returns the predicted counts.
    For the seq2PRINT model, the second output is the predicted counts.

    Parameters
    ----------
    model: torch.nn.Module
        A torch model to be wrapped.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, **kwargs):
        return self.model(X, **kwargs)[1][..., None]
