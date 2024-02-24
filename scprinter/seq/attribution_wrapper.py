import torch
import torch.nn as nn
from ..utils import zscore2pval_torch

class FootprintScaling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return zscore2pval_torch(x)

class _ProfileLogitScaling(torch.nn.Module):
    """This ugly class is necessary because of Captum.

    Captum internally registers classes as linear or non-linear. Because the
    profile wrapper performs some non-linear operations, those operations must
    be registered as such. However, the inputs to the wrapper are not the
    logits that are being modified in a non-linear manner but rather the
    original sequence that is subsequently run through the model. Hence, this
    object will contain all of the operations performed on the logits and
    can be registered.


    Parameters
    ----------
    logits: torch.Tensor, shape=(-1, -1)
        The logits as they come out of a Chrom/BPNet model.
    """

    def __init__(self):
        super(_ProfileLogitScaling, self).__init__()

    def forward(self, logits):
        weight = torch.exp(logits - torch.logsumexp(logits, dim=-1, keepdims=True)).detach()
        y = logits * weight
        # indices = torch.linspace(0, 1, logits.shape[-1], device=logits.device)
        # y = indices * weight * logits
        return y


class Smoother(nn.Module):
    def __init__(self, mode, decay):
        super().__init__()
        self.mode = mode
        self.decay = decay

        self.layer1 = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                      kernel_size=(99, 1), bias=False)
        # initialize:
        with torch.no_grad():
            v = decay ** (torch.abs(torch.arange(99) - mode))
            v = v / v.sum()
            self.layer1.weight.data[0, 0, :, 0] = v
        self.layer1.weight.requires_grad = False

    def forward(self, x):
        # x = x + torch.flip(x, dims=[3])
        x = self.layer1(x)
        return x

# This kinda mimics the way chrombpnet handles shape, but for sigmoid based stuff
class InverseSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ProfileWrapperFootprintClass_diff(torch.nn.Module):
    """A wrapper class that returns transformed profiles.
    This is for classification based models
        Parameters
        ----------
        model: torch.nn.Module
            A torch model to be wrapped.
        """

    def __init__(self, model, model_bg, nth_output=0,
                 reduce_mean=True,
                 specific_pos=None,
                 res=1, decay=None):
        super().__init__()
        self.model = model
        self.mode_bg = model_bg
        self.nth_output = nth_output
        self.res = res
        self.reduce_mean = reduce_mean
        self.scaling = InverseSigmoid()
        self.specific_pos = specific_pos
        self.decay = decay
        if decay is not None:
            print ("decay mode")
            self.decay = Smoother(self.nth_output, self.decay)

    def forward(self, X, *args, **kwargs):
        logits,_ = self.model(X, *args, **kwargs)
        logits_bg,_ = self.mode_bg(X, *args, **kwargs)
        logits = logits - logits_bg
        if self.specific_pos is not None:
            logits = logits[:, :, self.specific_pos]
        if self.decay is not None:
            logits = self.decay.forward(logits[:, None, :, :])[:, 0, :, :]
            logits = logits
        else:
            if self.nth_output is not None:
                logits = logits[:, self.nth_output]

        logits = logits.reshape(X.shape[0], -1)
        if self.reduce_mean:
            logits = logits - torch.mean(logits, dim=-1, keepdims=True)
        y = self.scaling(logits)
        # if self.decay is not None:
        #     # y = y.reshape(X.shape)
        #     # print (y.shape)
        #     y = self.decay.forward(y[:, None, :, :])
        #     # print (y.shape)
        #     y = y.reshape(X.shape[0], -1)

        y = y.sum(axis=-1, keepdims=True)
        return y

class ProfileWrapperFootprintClass(torch.nn.Module):
    """A wrapper class that returns transformed profiles.
    This is for classification based models
        Parameters
        ----------
        model: torch.nn.Module
            A torch model to be wrapped.
        """

    def __init__(self, model, nth_output=0,
                 reduce_mean=True,
                 specific_pos=None,
                 res=1, decay=None):
        super().__init__()
        self.model = model
        self.nth_output = nth_output
        self.res = res
        self.reduce_mean = reduce_mean
        self.scaling = InverseSigmoid()
        self.specific_pos = specific_pos
        self.decay = decay
        if decay is not None:
            print ("decay mode")
            self.decay = Smoother(self.nth_output, self.decay)

    def forward(self, X, *args, **kwargs):
        logits,_ = self.model(X, *args, **kwargs)
        if self.specific_pos is not None:
            logits = logits[:, :, self.specific_pos]
        if self.decay is not None:
            logits = self.decay.forward(logits[:, None, :, :])[:, 0, :, :]
            logits = logits
        else:
            if self.nth_output is not None:
                logits = logits[:, self.nth_output]
        shapes = logits.shape
        logits = logits.reshape(X.shape[0], -1)

        if self.reduce_mean:
            logits = logits - torch.mean(logits, dim=-1, keepdims=True)
        # logits = logits.reshape(shapes)
        #
        # if self.specific_pos is not None:
        #     logits = logits[:, :, self.specific_pos]

        logits = logits.reshape(X.shape[0], -1)
        y = self.scaling(logits)
        y = y.sum(axis=-1, keepdims=True)
        return y

# This is for a regression model, and uses chrombpnet way of predicting the shape
class ProfileWrapperFootprint(torch.nn.Module):
    """A wrapper class that returns transformed profiles.

        This class takes in a trained model and returns the weighted softmaxed
        outputs of the first dimension. Specifically, it takes the predicted
        "logits" and takes the dot product between them and the softmaxed versions
        of those logits. This is for convenience when using captum to calculate
        attribution scores.

        Parameters
        ----------
        model: torch.nn.Module
            A torch model to be wrapped.
        """

    def __init__(self, model, nth_output=0, res=1, reduce_mean=True,specific_pos=None,):
        super().__init__()
        self.model = model
        self.nth_output = nth_output
        self.res = res
        self.reduce_mean = reduce_mean
        self.scaling = _ProfileLogitScaling()
        self.specific_pos = specific_pos

    def forward(self, X, *args, **kwargs):
        logits,_ = self.model(X, *args, **kwargs)
        # logits = logits * (-1)
        if self.specific_pos is not None:
            logits = logits[:, :, self.specific_pos]
        if self.nth_output is not None:
            logits = logits[:, self.nth_output]
        logits = logits.reshape(X.shape[0], -1)
        if self.reduce_mean:
            logits = logits - torch.mean(logits, dim=-1, keepdims=True)
        y = self.scaling(logits)
        return (y).sum(axis=-1, keepdims=True)



class JustSumWrapper(torch.nn.Module):
    def __init__(self, model, nth_output=None, res=1, reduce_mean=True,
                 specific_pos=None, weight=None, threshold=None):
        super().__init__()
        self.model = model
        self.nth_output = nth_output
        self.res = res
        # self.reduce_mean = reduce_mean
        # self.scaling = _ProfileLogitScaling()
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
            logits = self.relu(logits-self.threshold)
        if self.weight is not None:
            logits = logits * self.weight

        # if self.reduce_mean:
        #     logits = logits - torch.mean(logits, dim=-1, keepdims=True)
        # y = self.scaling(logits)
        y = logits
        return (y).sum(axis=-1, keepdims=True)

class CountWrapper(torch.nn.Module):
    """A wrapper class that only returns the predicted counts.

    This class takes in a trained model and returns only the second output.
    For BPNet models, this means that it is only returning the count
    predictions. This is for convenience when using captum to calculate
    attribution scores.

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

