import warnings
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from packaging import version
from shap.explainers._explainer import Explainer
from torch.cuda.amp import GradScaler

"""
Code adapted from DeepSHAP, but modified with custom non-linear operations
"""


def _check_additivity(explainer, model_output_values, output_phis):
    TOLERANCE = 1e-2

    assert (
        len(explainer.expected_value) == model_output_values.shape[1]
    ), "Length of expected values and model outputs does not match."

    for l in range(len(explainer.expected_value)):
        if not explainer.multi_input:
            diffs = (
                model_output_values[:, l]
                - explainer.expected_value[l]
                - output_phis[l].sum(axis=tuple(range(1, output_phis[l].ndim)))
            )
        else:
            diffs = model_output_values[:, l] - explainer.expected_value[l]

            for i in range(len(output_phis[l])):
                diffs -= output_phis[l][i].sum(axis=tuple(range(1, output_phis[l][i].ndim)))

        maxdiff = torch.abs(diffs).max()

        assert maxdiff < TOLERANCE, (
            "The SHAP explanations do not sum up to the model's output! This is either because of a "
            "rounding error or because an operator in your computation graph was not fully supported. If "
            "the sum difference of %f is significant compared to the scale of your model outputs, please post "
            f"as a github issue, with a reproducible example so we can debug it. Used framework: {explainer.framework} - Max. diff: {maxdiff} - Tolerance: {TOLERANCE}"
        )


class PyTorchDeep(Explainer):

    def __init__(self, model, data):
        # try and import pytorch
        global torch
        if torch is None:
            import torch

            if version.parse(torch.__version__) < version.parse("0.4"):
                warnings.warn("Your PyTorch version is older than 0.4 and not supported.")

        # check if we have multiple inputs
        self.multi_input = False
        if isinstance(data, list):
            self.multi_input = True
        if not isinstance(data, list):
            data = [data]
        self.data = data
        self.layer = None
        self.input_handle = None
        self.interim = False
        self.interim_inputs_shape = None
        self.expected_value = None  # to keep the DeepExplainer base happy
        if type(model) == tuple:
            self.interim = True
            model, layer = model
            model = model.eval()
            self.layer = layer
            self.add_target_handle(self.layer)

            # if we are taking an interim layer, the 'data' is going to be the input
            # of the interim layer; we will capture this using a forward hook
            with torch.no_grad():
                _ = model(*data)
                interim_inputs = self.layer.target_input
                if type(interim_inputs) is tuple:
                    # this should always be true, but just to be safe
                    self.interim_inputs_shape = [i.shape for i in interim_inputs]
                else:
                    self.interim_inputs_shape = [interim_inputs.shape]
            self.target_handle.remove()
            del self.layer.target_input
        self.model = model.eval()

        self.multi_output = False
        self.num_outputs = 1
        with torch.no_grad():
            outputs = model(*data)

            # also get the device everything is running on
            self.device = outputs.device
            if outputs.shape[1] > 1:
                self.multi_output = True
                self.num_outputs = outputs.shape[1]
            self.expected_value = outputs.mean(0).cpu()
        self.amp = True

    def add_target_handle(self, layer):
        input_handle = layer.register_forward_hook(get_target_input)
        self.target_handle = input_handle

    def add_handles(self, model, forward_handle, backward_handle):
        """
        Add handles to all non-container layers in the model.
        Recursively for non-container layers
        """
        handles_list = []
        model_children = list(model.children())

        if hasattr(model, "shap_register"):
            handles_list.append(model.register_forward_hook(forward_handle))
            handles_list.append(model.register_full_backward_hook(backward_handle))
            return handles_list

        if model_children:
            # print ("has children", model)
            for child in model_children:
                handles_list.extend(self.add_handles(child, forward_handle, backward_handle))
        else:  # leaves
            # print("no children", model)
            # print (model, backward_handle)
            handles_list.append(model.register_forward_hook(forward_handle))
            handles_list.append(model.register_full_backward_hook(backward_handle))
        return handles_list

    def remove_attributes(self, model):
        """
        Removes the x and y attributes which were added by the forward handles
        Recursively searches for non-container layers
        """
        for child in model.children():
            if "nn.modules.container" in str(type(child)):
                self.remove_attributes(child)
            else:
                try:
                    del child.x
                except AttributeError:
                    pass
                try:
                    del child.y
                except AttributeError:
                    pass

    def gradient(self, idx, inputs, return_output=False):
        self.model.zero_grad()
        X = [x.requires_grad_() for x in inputs]

        scaler = GradScaler(enabled=self.amp)

        try:
            autocast_context = torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=self.amp
            )
        except RuntimeError:
            autocast_context = torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.amp
            )
        with autocast_context:
            outputs = self.model(*X)

        selected = [val for val in outputs[:, idx]]

        grads = []
        if self.interim:
            interim_inputs = self.layer.target_input
            for idx, input in enumerate(interim_inputs):

                scaled_grad = torch.autograd.grad(
                    selected,
                    input,
                    retain_graph=True if idx + 1 < len(interim_inputs) else None,
                    allow_unused=True,
                )[0]
                inv_scale = 1.0 / scaler.get_scale()
                grad = scaled_grad * inv_scale

                if grad is not None:
                    grad = grad.cpu()
                else:
                    grad = torch.zeros_like(X[idx]).cpu()
                grads.append(grad)
            del self.layer.target_input
            return grads, [i.detach().cpu() for i in interim_inputs]
        else:
            for idx, x in enumerate(X):
                # grad = torch.autograd.grad(selected, x,
                #                            retain_graph=True if idx + 1 < len(X) else None,
                #                            allow_unused=True)[0]
                scaled_grad = torch.autograd.grad(
                    outputs=selected,
                    inputs=x,
                    # create_graph=True,
                    retain_graph=True if idx + 1 < len(X) else None,
                    allow_unused=True,
                )[0]
                # inv_scale = 1. / scaler.get_scale()
                grad = scaled_grad  # * inv_scale
                # scaler.update()

                if grad is not None:
                    grad = grad.cpu()
                else:
                    grad = torch.zeros_like(X[idx]).cpu()
                grads.append(grad)
            if return_output:
                return grads, outputs.detach()
            else:
                return grads
            # return grads

    def shap_values(
        self,
        X,
        ranked_outputs=None,
        output_rank_order="max",
        check_additivity=True,
        custom_attribution_func=None,
    ):
        # X ~ self.model_input
        # X_data ~ self.data

        # check if we have multiple inputs
        if not self.multi_input:
            assert not isinstance(X, list), "Expected a single tensor model input!"
            X = [X]
        else:
            assert isinstance(X, list), "Expected a list of model inputs!"

        X = [x.detach().to(self.device) for x in X]

        model_output_values = None

        if ranked_outputs is not None and self.multi_output:
            with torch.no_grad():
                model_output_values = self.model(*X)
            # rank and determine the model outputs that we will explain
            if output_rank_order == "max":
                _, model_output_ranks = torch.sort(model_output_values, descending=True)
            elif output_rank_order == "min":
                _, model_output_ranks = torch.sort(model_output_values, descending=False)
            elif output_rank_order == "max_abs":
                _, model_output_ranks = torch.sort(torch.abs(model_output_values), descending=True)
            else:
                assert False, "output_rank_order must be max, min, or max_abs!"
            model_output_ranks = model_output_ranks[:, :ranked_outputs]
        else:
            model_output_ranks = (
                torch.ones((X[0].shape[0], self.num_outputs)).int()
                * torch.arange(0, self.num_outputs).int()
            )

        # add the gradient handles
        handles = self.add_handles(self.model, add_interim_values, deeplift_grad)
        if self.interim:
            self.add_target_handle(self.layer)

        # compute the attributions
        output_phis = []
        output_deltas = []
        # for the ith output of the model (usually only one output)
        for i in range(model_output_ranks.shape[1]):
            phis = []
            deltas = []
            if self.interim:
                for k in range(len(self.interim_inputs_shape)):
                    phis.append(torch.zeros((X[0].shape[0],) + self.interim_inputs_shape[k][1:]))
            else:
                # it assumes X is a list (multi-input)
                for k in range(len(X)):
                    phis.append(torch.zeros(X[k].shape))
                    deltas.append([])

            # for jth sample in X
            for j in range(X[0].shape[0]):
                # tile the inputs to line up with the background data samples
                tiled_X = [
                    X[l][j : j + 1].repeat(
                        (self.data[l].shape[0],) + tuple([1 for k in range(len(X[l].shape) - 1)])
                    )
                    for l in range(len(X))
                ]
                joint_x = [torch.cat((tiled_X[l], self.data[l]), dim=0) for l in range(len(X))]
                # run attribution computation graph
                feature_ind = model_output_ranks[j, i]
                sample_phis, outputs = self.gradient(feature_ind, joint_x, return_output=True)

                # assign the attributions to the right part of the output arrays
                if self.interim:
                    sample_phis, output = sample_phis
                    x, data = [], []
                    for k in range(len(output)):
                        x_temp, data_temp = torch.chunk(output[k], 2)
                        x.append(x_temp)
                        data.append(data_temp)
                    for l in range(len(self.interim_inputs_shape)):
                        phis[l][j] = (
                            sample_phis[l][self.data[l].shape[0] :] * (x[l] - data[l])
                        ).mean(0)
                else:
                    for l in range(len(X)):
                        output_diff = torch.sub(*torch.chunk(outputs[..., feature_ind], 2))
                        input_diff = torch.sum(
                            (tiled_X[l] - self.data[l])
                            * sample_phis[l][self.data[l].shape[0] :].to(self.device),
                            dim=(1, 2),
                        )
                        delta = output_diff - input_diff
                        deltas[l].append(delta.cpu().detach())
                        if custom_attribution_func is not None:
                            (attr,) = custom_attribution_func(
                                (sample_phis[l][self.data[l].shape[0] :].to(self.device),),
                                (tiled_X[l].to(self.device),),
                                (self.data[l].to(self.device),),
                            )
                            phis[l][j] = attr.mean(dim=0).cpu().detach()
                        else:
                            phis[l][j] = (
                                (
                                    sample_phis[l][self.data[l].shape[0] :].to(self.device)
                                    * (X[l][j : j + 1] - self.data[l])
                                )
                                .cpu()
                                .detach()
                                .mean(0)
                            )
            output_phis.append(phis[0] if not self.multi_input else phis)
            output_deltas.append(deltas[0] if not self.multi_input else deltas)
        # cleanup; remove all gradient handles
        for handle in handles:
            handle.remove()
        self.remove_attributes(self.model)
        if self.interim:
            self.target_handle.remove()

        # check that the SHAP values sum up to the model output
        if check_additivity:
            if model_output_values is None:
                with torch.no_grad():
                    model_output_values = self.model(*X)

            _check_additivity(self, model_output_values.cpu(), output_phis)

        if not self.multi_output:
            return output_phis[0], output_deltas[0]
        elif ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis, output_deltas


def deeplift_grad(module, grad_input, grad_output):
    """The backward hook which computes the deeplift
    gradient for an nn.Module
    """
    # first, get the module type
    module_type = module.__class__.__name__
    # first, check the module is supported
    if module_type in op_handler:
        # print(module_type, op_handler[module_type])
        if op_handler[module_type].__name__ not in ["passthrough", "linear_1d"]:
            return op_handler[module_type](module, grad_input, grad_output)
    else:
        print(f"unrecognized nn.Module: {module_type}")
        return grad_input


def add_interim_values(module, input, output):
    """The forward hook used to save interim tensors, detached
    from the graph. Used to calculate the multipliers
    """
    try:
        del module.x
    except AttributeError:
        pass
    try:
        del module.y
    except AttributeError:
        pass
    module_type = module.__class__.__name__
    if module_type in op_handler:
        func_name = op_handler[module_type].__name__
        # First, check for cases where we don't need to save the x and y tensors
        if func_name == "passthrough":
            pass
        else:
            # check only the 0th input varies
            for i in range(len(input)):
                if i != 0 and type(output) is tuple:
                    assert input[i] == output[i], "Only the 0th input may vary!"
            # if a new method is added, it must be added here too. This ensures tensors
            # are only saved if necessary
            if func_name in ["maxpool", "nonlinear_1d", "softmax"]:
                # only save tensors if necessary
                if type(input) is tuple:
                    setattr(module, "x", torch.nn.Parameter(input[0].detach()))
                else:
                    setattr(module, "x", torch.nn.Parameter(input.detach()))
                if type(output) is tuple:
                    setattr(module, "y", torch.nn.Parameter(output[0].detach()))
                else:
                    setattr(module, "y", torch.nn.Parameter(output.detach()))
            elif (
                func_name == "nonlinear_2d_elementwise"
                or func_name == "nonlinear_2d_elementwise_mul"
                or func_name == "nonlinear_2d_elementwise_matmul"
            ):
                if type(input) is tuple:
                    setattr(
                        module,
                        "x",
                        (
                            torch.nn.Parameter(input[0].detach()),
                            torch.nn.Parameter(input[1].detach()),
                        ),
                    )
                else:
                    setattr(module, "x", torch.nn.Parameter(input.detach()))
                if type(output) is tuple:
                    setattr(module, "y", torch.nn.Parameter(output[0].detach()))
                else:
                    setattr(module, "y", torch.nn.Parameter(output.detach()))


def get_target_input(module, input, output):
    """A forward hook which saves the tensor - attached to its graph.
    Used if we want to explain the interim outputs of a model
    """
    try:
        del module.target_input
    except AttributeError:
        pass
    setattr(module, "target_input", input)


def passthrough(module, grad_input, grad_output):
    """No change made to gradients"""
    return None


def maxpool(module, grad_input, grad_output):
    pool_to_unpool = {
        "MaxPool1d": torch.nn.functional.max_unpool1d,
        "MaxPool2d": torch.nn.functional.max_unpool2d,
        "MaxPool3d": torch.nn.functional.max_unpool3d,
    }
    pool_to_function = {
        "MaxPool1d": torch.nn.functional.max_pool1d,
        "MaxPool2d": torch.nn.functional.max_pool2d,
        "MaxPool3d": torch.nn.functional.max_pool3d,
    }
    delta_in = module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2) :]
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # we also need to check if the output is a tuple
    y, ref_output = torch.chunk(module.y, 2)
    cross_max = torch.max(y, ref_output)
    diffs = torch.cat([cross_max - ref_output, y - cross_max], 0)

    # all of this just to unpool the outputs
    with torch.no_grad():
        _, indices = pool_to_function[module.__class__.__name__](
            module.x,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.ceil_mode,
            True,
        )
        xmax_pos, rmax_pos = torch.chunk(
            pool_to_unpool[module.__class__.__name__](
                grad_output[0] * diffs,
                indices,
                module.kernel_size,
                module.stride,
                module.padding,
                list(module.x.shape),
            ),
            2,
        )

    grad_input = [None for _ in grad_input]
    grad_input[0] = torch.where(
        torch.abs(delta_in) < 1e-7,
        torch.zeros_like(delta_in),
        (xmax_pos + rmax_pos) / delta_in,
    ).repeat(dup0)

    return tuple(grad_input)


def linear_1d(module, grad_input, grad_output):
    """No change made to gradients."""
    return None


def nonlinear_1d(module, grad_input, grad_output):

    delta_out = module.y[: int(module.y.shape[0] / 2)] - module.y[int(module.y.shape[0] / 2) :]
    delta_in = module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2) :]

    #     a,b = torch.chunk(delta_in, 2, dim=0)
    #     delta_in = a + b
    #     a,b = torch.chunk(grad_input[0], 2, dim=0)
    #     grad_input = (a + b,)
    # print(delta_out.shape, delta_in.shape)
    # print(grad_input[0].shape, grad_output[0].shape)
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # handles numerical instabilities where delta_in is very small by
    # just taking the gradient in those cases
    grads = [None for _ in grad_input]
    grads[0] = torch.where(
        torch.abs(delta_in.repeat(dup0)) < 1e-6,
        grad_input[0],
        grad_output[0] * (delta_out / delta_in).repeat(dup0),
    ).type(grad_input[0].dtype)

    return tuple(grads)


def nonlinear_2d_elementwise_mul(module, grad_input, grad_output):
    return nonlinear_2d_elementwise(module, grad_input, grad_output, torch.mul)


def nonlinear_2d_elementwise_matmul(module, grad_input, grad_output):
    return nonlinear_2d_elementwise(module, grad_input, grad_output, torch.matmul)


def nonlinear_2d_elementwise(module, grad_input, grad_output, operation):
    xout, rout = torch.chunk(module.y, 2)
    in0, in1 = module.x
    xin0, rin0 = torch.chunk(in0, 2)
    xin1, rin1 = torch.chunk(in1, 2)
    delta_in0 = xin0 - rin0
    delta_in1 = xin1 - rin1
    dup0 = [2] + [1 for i in delta_in0.shape[1:]]
    out10 = operation(xin0, rin1)
    out01 = operation(rin0, xin1)
    out11, out00 = xout, rout
    out0 = 0.5 * (out11 - out01 + out10 - out00)
    out0 = grad_output[0] * (out0 / delta_in0).repeat(dup0)
    out1 = 0.5 * (out11 - out10 + out01 - out00)
    out1 = grad_output[0] * (out1 / delta_in1).repeat(dup0)

    grads0 = [None for _ in grad_input]
    grads0[0] = torch.where(torch.abs(delta_in0.repeat(dup0)) < 1e-7, torch.zeros_like(out0), out0)

    grads1 = [None for _ in grad_input]
    grads1[0] = torch.where(torch.abs(delta_in1.repeat(dup0)) < 1e-7, torch.zeros_like(out1), out1)

    grads = [grads0[0], grads1[0]]
    return tuple(grads)


def softmax(module, grad_input, grad_output):
    delta_out = module.y[: int(module.y.shape[0] / 2)] - module.y[int(module.y.shape[0] / 2) :]
    delta_in = module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2) :]

    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # handles numerical instabilities where delta_in is very small by
    # just taking the gradient in those cases
    grads = [None for _ in grad_input]
    grad_input_unnorm = torch.where(
        torch.abs(delta_in.repeat(dup0)) < 1e-6,
        grad_input[0],
        grad_output[0] * (delta_out / delta_in).repeat(dup0),
    )

    # n = grad_input[0].shape[-1]
    n = grad_input[0].numel()

    # norm = grad_input_unnorm.sum(axis=-1) * 1 / n
    # norm = norm.view(norm.shape + (1,))
    #
    # # updating only the first half
    # grads[0] = grad_input_unnorm - norm
    grads[0] = grad_input_unnorm - grad_input_unnorm.sum() * 1 / n
    return tuple(grads)


# import torch

# def softmax(module, grad_input, grad_output):
#
#     with torch.autograd.set_grad_enabled(True):
#         # print(grad_output[0].shape, grad_output[0].max())
#
#         in0 = module.x.clone().detach()  # Assuming 'module.x' is the input tensor like in 'nonlinear_1d'
#         in0_max = torch.max(in0, dim=-1, keepdim=True)[0]
#         in0_centered = in0 - in0_max
#         in0_centered = in0_centered.requires_grad_()
#         evals = torch.exp(in0_centered)
#         rsum = torch.sum(evals, dim=-1, keepdim=True) + 1
#         div = evals / rsum
#
#         grad = torch.autograd.grad(div, in0_centered, grad_output[0],
#                                    allow_unused=True)[0].detach()
#     # print (torch.allclose(grad, grad_input[0]))
#     # print (grad[0], grad_input[0][0], grad_output[0][0])
#     delta_in = in0[: in0.size(0) // 2] - in0[in0.size(0) // 2:]
#     delta_in_centered = in0_centered[: in0_centered.size(0) // 2] - in0_centered[in0_centered.size(0) // 2:]
#     delta_out = module.y[: int(module.y.shape[0] / 2)] - module.y[int(module.y.shape[0] / 2):]
#     dup0 = [2] + [1 for _ in delta_in.shape[1:]]
#     grads = [None for _ in grad_input]
#     # print (delta_in_centered, delta_in)
#
#     grads[0] = torch.where(torch.abs(delta_in.repeat(dup0)) < 1e-6, grad,
#                            grad * (delta_in_centered / delta_in).repeat(dup0))
#     # grads[0] = torch.where(torch.abs(delta_in_centered.repeat(dup0)) < 1e-6, grad,
#     #                        grad_output[0] * (delta_out / delta_in_centered).repeat(dup0))
#     # compare = nonlinear_1d(module, grad_input, grad_output)
#     # return compare
#     # print ((grads[0].max(), compare[0].max()))
#     return tuple(grads)
#

op_handler = {}

# passthrough ops, where we make no change to the gradient
op_handler["Dropout3d"] = passthrough
op_handler["Dropout2d"] = passthrough
op_handler["Dropout"] = passthrough
op_handler["AlphaDropout"] = passthrough

op_handler["Conv1d"] = linear_1d
op_handler["Conv2d"] = linear_1d
op_handler["Conv3d"] = linear_1d
op_handler["ConvTranspose1d"] = linear_1d
op_handler["ConvTranspose2d"] = linear_1d
op_handler["ConvTranspose3d"] = linear_1d
op_handler["Linear"] = linear_1d
op_handler["AvgPool1d"] = linear_1d
op_handler["AvgPool2d"] = linear_1d
op_handler["AvgPool3d"] = linear_1d
op_handler["AdaptiveAvgPool1d"] = linear_1d
op_handler["AdaptiveAvgPool2d"] = linear_1d
op_handler["AdaptiveAvgPool3d"] = linear_1d
op_handler["BatchNorm1d"] = linear_1d
op_handler["LayerNorm"] = linear_1d
op_handler["BatchNorm2d"] = linear_1d
op_handler["BatchNorm3d"] = linear_1d
op_handler["Rearrange"] = linear_1d
op_handler["Upsample"] = linear_1d
op_handler["Identity"] = linear_1d
op_handler["LeakyReLU"] = nonlinear_1d
op_handler["ReLU"] = nonlinear_1d
op_handler["GELU"] = nonlinear_1d
op_handler["ELU"] = nonlinear_1d
op_handler["Sigmoid"] = nonlinear_1d
op_handler["Tanh"] = nonlinear_1d
op_handler["Softplus"] = nonlinear_1d
op_handler["Softmax"] = linear_1d
op_handler["Softmax_one"] = nonlinear_1d
op_handler["SigmoidScale"] = nonlinear_1d
op_handler["InverseSigmoid"] = nonlinear_1d
op_handler["_ProfileLogitScaling"] = nonlinear_1d
op_handler["FootprintScaling"] = nonlinear_1d
op_handler["RelMultiHeadAttention_custom"] = passthrough
op_handler["ElementWiseMultiply"] = nonlinear_2d_elementwise_mul
op_handler["MatrixMultiply"] = linear_1d
op_handler["MaxPool1d"] = maxpool
op_handler["MaxPool2d"] = maxpool
op_handler["MaxPool3d"] = maxpool
