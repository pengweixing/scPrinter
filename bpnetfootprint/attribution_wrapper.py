import torch

from .utils import DNA_one_hot


### Haven't been cleaned
class SignalWrapperFootprint(torch.nn.Module):
    def __init__(self, model, nth_output=0, activation=None, transformation=None, res=1):
        super().__init__()
        self.model = model
        self.nth_output = nth_output
        self.activation = activation
        self.transformation = transformation
        self.res = res
        self.sniff_padding("N" * 2114)

    @torch.no_grad()
    def sniff_padding(self, X):
        dev = next(self.model.parameters()).device
        if type(X) is str:
            X = DNA_one_hot(X.upper())
        X = X.float().to(dev)[None]
        logits = self.model(X)[:, self.nth_output]
        # print (X.shape, logits.shape)
        self.padding = (X.shape[-1] - logits.shape[-1] * self.res) // 2

    def forward(self, X, *args, **kwargs):
        dev = next(self.model.parameters()).device
        if type(X) is str:
            X = DNA_one_hot(X.upper())
        X = X.float().to(dev)[None]
        logits = self.model(X, *args, **kwargs)[:, self.nth_output]
        if self.activation is not None:
            logits = self.activation(logits)
        if self.transformation is not None:
            w, b = self.transformation
            logits = logits * w + b
        logits = logits[0]
        return logits


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
        y = torch.nn.functional.log_softmax(logits, dim=-1)
        y = logits * torch.exp(y).detach()
        return y


# This kinda mimics the way chrombpnet handles shape, but for sigmoid based stuff
class InverseSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ProfileWrapperFootprintClass(torch.nn.Module):
    """A wrapper class that returns transformed profiles.
    This is for classification based models
        Parameters
        ----------
        model: torch.nn.Module
            A torch model to be wrapped.
    """

    def __init__(self, model, nth_output=0, reduce_mean=True, res=1):
        super().__init__()
        self.model = model
        self.nth_output = nth_output
        self.res = res
        self.reduce_mean = reduce_mean
        self.scaling = InverseSigmoid()

    def forward(self, X, *args, **kwargs):
        logits = self.model(X, *args, **kwargs)[:, self.nth_output]
        logits = logits.reshape(X.shape[0], -1)
        if self.reduce_mean:
            logits = logits - torch.mean(logits, dim=-1, keepdims=True)
        y = self.scaling(logits)
        return (y).mean(axis=-1, keepdims=True)


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

    def __init__(self, model, nth_output=0, res=1, reduce_mean=True):
        super().__init__()
        self.model = model
        self.nth_output = nth_output
        self.res = res
        self.reduce_mean = reduce_mean
        self.scaling = _ProfileLogitScaling()

    def forward(self, X, *args, **kwargs):
        logits = self.model(X, *args, **kwargs)[:, self.nth_output]
        logits = logits.reshape(X.shape[0], -1)
        if self.reduce_mean:
            logits = logits - torch.mean(logits, dim=-1, keepdims=True)
        y = self.scaling(logits)
        return (y).sum(axis=-1, keepdims=True)

    # def attribute(self, seq, additional_forward_args=None):
    #     dev = next(self.model.parameters()).device
    #     if type(seq) is str:
    #         seq = DNA_one_hot(seq.upper())
    #     _X = seq.float()
    #     _references = dinucleotide_shuffle(_X, n_shuffles=20).to(dev)
    #     _references = _references.type(_X.dtype)
    #     _X = _X.unsqueeze(0).to(dev)
    #     dl = DeepLiftShap(self)
    #     attr = dl.attribute(_X, _references,
    #                         custom_attribution_func=hypothetical_attributions,
    #                         additional_forward_args=(additional_forward_args, )
    #                         )
    #     attr = (attr * _X)[0].sum(dim=0).detach()
    #     return attr


class ProfileWrapper(torch.nn.Module):
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

    def __init__(self, model):
        super(ProfileWrapper, self).__init__()
        self.model = model
        self.scaling = _ProfileLogitScaling()

    def forward(self, X, **kwargs):
        logits = self.model(X, **kwargs)[0]
        logits = logits.reshape(X.shape[0], -1)
        logits = logits - torch.mean(logits, dim=-1, keepdims=True)

        y = self.scaling(logits)
        return y.sum(axis=-1, keepdims=True)


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
        super(CountWrapper, self).__init__()
        self.model = model

    def forward(self, X, **kwargs):
        return self.model(X, **kwargs)[1]
