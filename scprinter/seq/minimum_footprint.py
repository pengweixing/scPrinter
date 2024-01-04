'''
This contains the code for the minimum footprint algorithm in pytorch
Due to numerical stability issue, it's not going to replicate the original results
But it should be accurate enough to train a neural network
'''
import torch
import torch.nn as nn





def footprintWindowSum(x,  # A numerical or integer vector x
                       footprintRadius,  # Radius of the footprint region
                       flankRadius  # Radius of the flanking region (not including the footprint region)
                       ):
    halfFlankRadius = int(flankRadius / 2)
    width = x.shape[-1]

    # Calculate sum of x in the left flanking window
    shift = halfFlankRadius + footprintRadius
    # x can be shape of (x) or (sample, x)
    shapes = list(x.shape)
    shapes[-1] = shift
    leftShifted = torch.cat([torch.zeros(shapes, device=x.device), x], dim=-1)

    leftFlankSum = rz_conv_torch(leftShifted, halfFlankRadius)[..., :width]

    # Calculate sum of x in the right flanking window
    rightShifted = torch.cat([x, torch.zeros(shapes, device=x.device),], dim=-1)
    rightFlankSum = rz_conv_torch(rightShifted, halfFlankRadius)[..., shift:]

    centerSum = rz_conv_torch(x, footprintRadius)
    return leftFlankSum,centerSum,rightFlankSum

# Doing the same thing as conv in R, but more generalizable and in torch
def rz_conv_torch(a, n=2):
    if n == 0:
        return a
    shapes = a.shape
    if len(shapes) == 1:
        a = a[None, None, :]
    else:
        a = a.reshape((-1, 1, shapes[-1]))

    a = torch.nn.functional.conv1d(a,
                               torch.ones((1, 1, n * 2), device=a.device, dtype=a.dtype),
                               torch.zeros((1), device=a.device, dtype=a.dtype), padding=n)[...,0, 1:]

    if len(shapes) == 1:
        a = a[0]
    else:
        a = a.reshape(shapes)
    return a

def post_processing(xx, radius):
    # mask1 = torch.isnan(xx)
    # xx[mask1] = -float('inf')
    xx = torch.nn.functional.max_pool1d(xx, 2*radius, stride=1, padding=radius)[..., 1:]
    # xx[mask1] = float('nan')
    xx = rz_conv_torch(xx, radius) / (2 * radius)
    return xx

class RobustScale(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.model = nn.Linear(weight.shape[1], weight.shape[0])
        self.model.weight.data = weight
        self.model.bias.data = bias

    def forward(self, x):
        # mask1 = torch.isnan(x)
        # mask2 = torch.isinf(x)
        # sign = torch.sign(x)
        # x[mask1] = 0
        # x[mask2] = 0
        x = self.model(x)
        # x[mask1] = np.nan
        # x[mask2] = np.inf * sign[mask2]
        return x


class dispModel(nn.Module):
    def __init__(self, dispmodels=None):
        super().__init__()
        if dispmodels is not None:
            self.initialize(dispmodels)


    def initialize(self, dispmodels):
        self.models = []
        # self.scales2 = []

        self.scale1_weight_all = []
        self.scale1_bias_all = []
        self.linear1_weight_all = []
        self.linear1_bias_all = []
        self.linear2_weight_all = []
        self.linear2_bias_all = []
        self.scale2_weight_all = []
        self.scale2_bias_all = []


        for i in range(2, 101, 1):
            key = str(i)
            scale1 = RobustScale(torch.diag(1 / torch.from_numpy(dispmodels[key]['featureSD'])).float(),
                                 -torch.from_numpy(dispmodels[key]['featureMean']).float() / torch.from_numpy(
                                     dispmodels[key]['featureSD']).float())
            # print (scale1.weight.data, scale1.bias.data)

            scale2 = RobustScale(torch.diag(torch.from_numpy(dispmodels[key]['targetSD'])).float(),
                                 torch.from_numpy(dispmodels[key]['targetMean']).float())

            linear1 = nn.Linear(dispmodels[key]['modelWeights'][0].shape[1],
                                dispmodels[key]['modelWeights'][0].shape[0])
            linear1.weight.data = dispmodels[key]['modelWeights'][0].float()
            linear1.bias.data = dispmodels[key]['modelWeights'][1].float()

            linear2 = nn.Linear(dispmodels[key]['modelWeights'][2].shape[1],
                                dispmodels[key]['modelWeights'][2].shape[0])
            linear2.weight.data = dispmodels[key]['modelWeights'][2].float()
            linear2.bias.data = dispmodels[key]['modelWeights'][3].float()

            self.scale1_weight_all.append(scale1.model.weight.data.T)
            self.scale1_bias_all.append(scale1.model.bias.data[None])
            self.linear1_weight_all.append(linear1.weight.data.T)
            self.linear1_bias_all.append(linear1.bias.data[None])
            self.linear2_weight_all.append(linear2.weight.data.T)
            self.linear2_bias_all.append(linear2.bias.data[None])
            self.scale2_weight_all.append(scale2.model.weight.data.T)
            self.scale2_bias_all.append(scale2.model.bias.data[None])


            self.models.append(nn.Sequential(
                scale1,
                linear1,
                nn.ReLU(),
                linear2,
                scale2
            ))
        self.models = nn.ModuleList(self.models)

        self.scale1_weight_all = nn.Parameter(torch.stack(self.scale1_weight_all, dim=0))
        self.scale1_bias_all = nn.Parameter(torch.stack(self.scale1_bias_all, dim=0))
        self.linear1_weight_all = nn.Parameter(torch.stack(self.linear1_weight_all, dim=0))
        self.linear1_bias_all = nn.Parameter(torch.stack(self.linear1_bias_all, dim=0))
        self.linear2_weight_all = nn.Parameter(torch.stack(self.linear2_weight_all, dim=0))
        self.linear2_bias_all = nn.Parameter(torch.stack(self.linear2_bias_all, dim=0))
        self.scale2_weight_all = nn.Parameter(torch.stack(self.scale2_weight_all, dim=0))
        self.scale2_bias_all = nn.Parameter(torch.stack(self.scale2_bias_all, dim=0))

        for param in self.parameters():
            param.requires_grad = False

        self.smooth_kernels = torch.zeros((99, 1, 100)).float()
        for i in range(99):
            size = int((i+2) / 2)
            self.smooth_kernels[ i,:, 50-size:50+size] = 1
        self.smooth_kernels = nn.Parameter(self.smooth_kernels, requires_grad=False)

        self.smooth_kernels_center = torch.zeros((99, 1, 100 * 2)).float()
        for i in range(99):
            size = i+2
            self.smooth_kernels_center[i,:, 100-size:100 + size] = 1
        self.smooth_kernels = nn.Parameter(self.smooth_kernels, requires_grad=False)
        self.smooth_kernels_center = nn.Parameter(self.smooth_kernels_center, requires_grad=False)

    def windowsum(self, x, modes):
        shapes = list(x.shape)
        width = shapes[-1]

        if len(shapes) == 1:
            x = x[None, None, :]
        else:
            x = x.reshape((-1, 1, shapes[-1]))

        smooth_kernels = self.smooth_kernels[modes-2]
        smooth_kernels_center = self.smooth_kernels_center[modes-2]

        a = torch.nn.functional.conv1d(x,
                                       smooth_kernels,
                                       torch.zeros((len(modes)),
                                        device=x.device,
                                       dtype=x.dtype), padding=200)[..., 1:]
        # Now we need to slice the tensor
        l = []
        r = []
        for i, mode in enumerate(modes):
            ori_padding = int(mode / 2) + mode
            extra = 150 - ori_padding
            l.append(
                a[..., i, extra:extra+width]
            )
            if extra == 0:
                r.append(
                    a[..., i, -extra-width:]
                )
            else:
                r.append(
                    a[..., i, -extra-width:-extra]
                )
        l = torch.stack(l, dim=-2)
        r = torch.stack(r, dim=-2)

        c = torch.nn.functional.conv1d(x,
                                       smooth_kernels_center,
                                       torch.zeros((len(modes)),
                                                   device=x.device,
                                                   dtype=x.dtype), padding=100)[..., :, 1:]
        if len(shapes) == 1:
            l = l[0]
            c = c[0]
            r = r[0]
        else:
            shapes = shapes[:-1] + [len(modes), width]
            l = l.reshape(shapes)
            c = c.reshape(shapes)
            r = r.reshape(shapes)

        return l, c, r


    def footprint(self, atac, bias, modes,
                   clip_min=-10,
                   clip_max=10):
        bias_and_atac = torch.cat([bias, atac], dim=0)
        bulk_l, bulk_c, bulk_r = self.windowsum(bias_and_atac, modes)
        biasWindowSums_l_bulk, biasWindowSums_c_bulk, biasWindowSums_r_bulk = (bulk_l[:len(bias)],
                                                                               bulk_c[:len(bias)],
                                                                               bulk_r[:len(bias)])
        insertionWindowSums_l_bulk, insertionWindowSums_c_bulk, insertionWindowSums_r_bulk = (bulk_l[len(bias):],
                                                                                              bulk_c[len(bias):],
                                                                                              bulk_r[len(bias):])
        leftTotalInsertion_bulk = insertionWindowSums_c_bulk + insertionWindowSums_l_bulk
        rightTotalInsertion_bulk = insertionWindowSums_c_bulk + insertionWindowSums_r_bulk
        fgFeatures_bulk = torch.stack([
            biasWindowSums_l_bulk, biasWindowSums_r_bulk,
            biasWindowSums_c_bulk,
            torch.log10(leftTotalInsertion_bulk),
            torch.log10(rightTotalInsertion_bulk)
        ], dim=-1)  # shape of [..., scales, length, 5]
        shapes = list(fgFeatures_bulk.shape)
        num_dims = len(shapes)
        # Create a tuple representing the new order of dimensions
        # The first (num_dims - 3) dimensions remain unchanged
        new_order = tuple(range(num_dims - 3)) + (num_dims - 2, num_dims - 3, num_dims - 1)
        fgFeatures_bulk = fgFeatures_bulk.permute(*new_order) # shapes of [...., length, scales, 5]
        # Now do batch matmul

        scale1_weights = self.scale1_weight_all[modes-2] # shape of [scales, input, output]
        scale1_bias = self.scale1_bias_all[modes-2] # shape of [scales, output]
        linear1_weights = self.linear1_weight_all[modes-2]
        linear1_bias = self.linear1_bias_all[modes-2]
        linear2_weights = self.linear2_weight_all[modes-2]
        linear2_bias = self.linear2_bias_all[modes-2]
        scale2_weights = self.scale2_weight_all[modes-2]
        scale2_bias = self.scale2_bias_all[modes-2]

        output = torch.matmul(fgFeatures_bulk[..., None, :], scale1_weights[None, None]) + scale1_bias
        output = torch.matmul(output, linear1_weights[None, None]) + linear1_bias
        output = torch.nn.functional.relu(output)
        output = torch.matmul(output, linear2_weights[None, None]) + linear2_bias
        output = torch.matmul(output, scale2_weights[None, None]) + scale2_bias # shape of [..., length, scales, 1, 4]
        # output = output.reshape(shapes[:-1] + [4]) # [..., scale, 4]
        predDispersion = output[..., 0, :].permute(*new_order) # [..., scales, length, 4]

        leftPredRatioMean = predDispersion[..., 0] # shape of [..., scales, length]
        leftPredRatioSD = predDispersion[..., 1]
        rightPredRatioMean = predDispersion[..., 2]
        rightPredRatioSD = predDispersion[..., 3]

        leftPredRatioSD[leftPredRatioSD < 0] = 0
        rightPredRatioSD[rightPredRatioSD < 0] = 0

        fgLeftRatio = insertionWindowSums_c_bulk / leftTotalInsertion_bulk # shape of [..., scales, length]
        fgRightRatio = insertionWindowSums_c_bulk / rightTotalInsertion_bulk

        leftZ = (fgLeftRatio - leftPredRatioMean) / leftPredRatioSD
        rightZ = (fgRightRatio - rightPredRatioMean) / (rightPredRatioSD)
        p = torch.maximum(leftZ, rightZ)

        # Mask positions with zero coverage on either flanking side
        p[(leftTotalInsertion_bulk < 1) | (rightTotalInsertion_bulk < 1) | (leftPredRatioSD == 0) | (
                    rightPredRatioSD == 0)] = float('nan')
        zs = p
        mask = torch.isinf(zs)
        zs[mask] = float('nan')
        zs.clamp_(clip_min, clip_max)
        zs = zs[..., 100:-100]
        return zs.float()


    def forward(self, x: torch.Tensor, i: int):
        x = self.models[i - 2](x)
        return x



@torch.no_grad()
def multiscaleFoot_new(atac,
                   bias,
                   modes,
                   dispmodel,
                   clip_min=-10,
                   clip_max=10):
    atac = atac.float()
    bias = bias.float()
    return dispmodel.footprint(atac, bias, modes, clip_min=clip_min, clip_max=clip_max)

@torch.no_grad()
def multiscaleFoot_old(atac,
                   bias,
                   modes,
                   dispmodel,
                   clip_min=-10,
                   clip_max=10):

    zs = []
    atac = atac.float()
    bias = bias.float()

    for i, mode in enumerate(modes):
        footprintRadius = mode
        flankRadius = mode
        # Get sum of predicted bias in left flanking, center, and right flanking windows
        biasWindowSums_l, biasWindowSums_c, biasWindowSums_r = footprintWindowSum(bias,
                                                                                  footprintRadius,
                                                                                  flankRadius)


        # Get sum of insertion counts in left flanking, center, and right flanking windows
        insertionWindowSums_l, insertionWindowSums_c, insertionWindowSums_r = footprintWindowSum(atac,
                                                                                                 footprintRadius,
                                                                                                 flankRadius)



        leftTotalInsertion = insertionWindowSums_c + insertionWindowSums_l
        rightTotalInsertion = insertionWindowSums_c + insertionWindowSums_r
        fgFeatures = torch.stack([
            biasWindowSums_l, biasWindowSums_r,
            biasWindowSums_c,
            torch.log10(leftTotalInsertion),
            torch.log10(rightTotalInsertion)
        ], dim=-1)


        predDispersion = dispmodel(fgFeatures, mode)

        leftPredRatioMean = predDispersion[..., 0]
        leftPredRatioSD = predDispersion[..., 1]
        rightPredRatioMean = predDispersion[..., 2]
        rightPredRatioSD = predDispersion[..., 3]

        leftPredRatioSD[leftPredRatioSD < 0] = 0
        rightPredRatioSD[rightPredRatioSD < 0] = 0

        fgLeftRatio = insertionWindowSums_c / leftTotalInsertion
        fgRightRatio = insertionWindowSums_c / rightTotalInsertion

        leftZ = (fgLeftRatio - leftPredRatioMean) / leftPredRatioSD
        rightZ = (fgRightRatio - rightPredRatioMean) / (rightPredRatioSD)

        p = torch.maximum(leftZ, rightZ)

        # Mask positions with zero coverage on either flanking side
        p[(leftTotalInsertion < 1) | (rightTotalInsertion < 1) | (leftPredRatioSD == 0) | (rightPredRatioSD == 0)] = float('nan')
        zs.append(p)

    zs = torch.stack(zs, axis=1)
    mask = torch.isinf(zs)
    zs[mask] = float('nan')
    zs.clamp_(clip_min, clip_max)
    zs = zs[...,100:-100]

    return zs.float()


multiscaleFoot = multiscaleFoot_new