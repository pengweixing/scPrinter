import argparse
import os
import pickle
import random
import re

import numpy as np
import pandas as pd
import pyBigWig
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm, trange

import scprinter as scp


# Validation step with AUPR reporting
def validate_model(model, val_loader, criterion, device="cpu"):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No need to compute gradient when evaluating
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[..., 0, :]
            val_loss += criterion(outputs, labels.float()).item()

            # # Store predictions and labels
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())  # Assuming binary classification
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_loss *= 100
    val_aupr = average_precision_score(all_labels, all_preds)  # Compute AUPR
    return val_loss, val_aupr


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import *

from scprinter.seq.ema import EMA


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        else:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# 1. Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, input_channel, mean, scale):
        super(MLP, self).__init__()
        self.mean = nn.Parameter(torch.from_numpy(mean).float())
        self.scale = nn.Parameter(torch.from_numpy(scale).float())
        self.fc1 = nn.Linear((input_size) * input_channel, 256)  # First hidden layer
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 128)  # Second hidden layer
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(128, 64)  # Output layer, assuming 10 classes
        self.activation3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(64, 1)  # Output layer, assuming 10 classes

        # self.fc5 = nn.Linear(3, 1)

    def forward(self, x):
        x = (x - self.mean[-x.shape[-1] :]) / self.scale[-x.shape[-1] :]
        xx = self.dropout1(self.activation1(self.fc1(x[..., :].reshape(x.shape[0], -1))))
        xx = self.dropout2(self.activation2(self.fc2(xx)))
        xx = self.dropout3(self.activation3(self.fc3(xx)))
        xx = self.fc4(xx)  # No activation, will use nn.CrossEntropyLoss
        # xx = xx + self.fc5(x[..., :3].mean(dim=1))
        return xx


class TFConv(nn.Module):
    def __init__(self, input_size, input_channel, mean, scale):
        super(TFConv, self).__init__()
        self.mean = nn.Parameter(torch.from_numpy(mean).float())
        self.scale = nn.Parameter(torch.from_numpy(scale).float())
        self.mean.requires_grad = False
        self.scale.requires_grad = False
        self.input_size = input_size
        self.input_channel = input_channel

        self.conv1 = nn.Conv1d(input_channel, 256, input_size)
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv1d(256, 128, 1)
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv1d(128, 64, 1)
        self.activation3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.25)
        self.conv4 = nn.Conv1d(64, 1, 1)

    def forward(self, x):

        if x.shape[-1] != self.mean.shape[-1]:
            x = (x - self.mean[3:]) / self.scale[3:]
            x = torch.cat(
                [
                    torch.zeros([x.shape[0], x.shape[1], 3], device=x.device, dtype=x.dtype),
                    x,
                ],
                dim=2,
            )
        else:
            x = (x - self.mean) / self.scale
        # x = (x - self.mean) / self.scale
        xx = self.dropout1(self.activation1(self.conv1(x)))
        xx = self.dropout2(self.activation2(self.conv2(xx)))
        xx = self.dropout3(self.activation3(self.conv3(xx)))
        xx = self.conv4(xx)
        return xx


from copy import deepcopy


class TFConv_translate(nn.Module):
    def __init__(self, tfconv):
        super(TFConv_translate, self).__init__()
        # self = deepcopy(tfconv)
        print(
            tfconv.conv1.weight.shape,
            tfconv.conv1.bias.shape,
            tfconv.mean.shape,
            tfconv.scale.shape,
        )
        new_weight_0 = 1 / tfconv.scale[None] * tfconv.conv1.weight[:, 0, :]
        new_weight_1 = (
            tfconv.conv1.bias - (tfconv.mean / tfconv.scale) @ tfconv.conv1.weight[:, 0, :].T
        )
        print(new_weight_0.shape, new_weight_1.shape)

        self.conv1 = nn.Conv1d(tfconv.input_channel, 256, tfconv.input_size)
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv1d(256, 128, 1)
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv1d(128, 64, 1)
        self.activation3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.25)
        self.conv4 = nn.Conv1d(64, 1, 1)

        self.conv1.weight.data = new_weight_0[:, None, :]
        self.conv1.bias.data = new_weight_1
        self.with_motif = True
        self.conv2.weight.data = tfconv.conv2.weight.data
        self.conv2.bias.data = tfconv.conv2.bias.data
        self.conv3.weight.data = tfconv.conv3.weight.data
        self.conv3.bias.data = tfconv.conv3.bias.data
        self.conv4.weight.data = tfconv.conv4.weight.data
        self.conv4.bias.data = tfconv.conv4.bias.data

    def with_motif(self, flag):
        self.with_motif = flag

    def forward(self, x):
        # if not self.with_motif:
        #     x = F.conv1d(x, self.conv1.weight[:, :, 3:], self.conv1.bias)
        # else:
        x = F.conv1d(x, self.conv1.weight, self.conv1.bias)

        xx = self.dropout1(self.activation1(x))
        xx = self.dropout2(self.activation2(self.conv2(xx)))
        xx = self.dropout3(self.activation3(self.conv3(xx)))
        xx = self.conv4(xx)
        return xx


# The model is now trained, you\ can save it or further evaluate it.


torch.set_num_threads(4)


parser = argparse.ArgumentParser(description="Bindingscore BPNet")
parser.add_argument(
    "--feats",
    nargs="+",
    type=int,
)

args = parser.parse_args()
feats = args.feats
feats = np.array(feats)

train_feats, valid_feats, train_labels, valid_labels = pickle.load(open("finetune_TFBS.pkl", "rb"))
train_feats = train_feats[:, feats, 3:]
train_shape = train_feats.shape
train_feats = train_feats.reshape((len(train_feats), len(feats), -1))
valid_feats = valid_feats[:, feats, 3:]
valid_shape = valid_feats.shape
valid_feats = valid_feats.reshape((len(valid_feats), len(feats), -1))


from sklearn.preprocessing import *

scaler = StandardScaler().fit(train_feats[:, 0, :])
mean, std = scaler.mean_, scaler.scale_

# train_feats = scaler.transform(train_feats).reshape(train_shape)
# valid_feats = scaler.transform(valid_feats).reshape(valid_shape)


X_train = torch.as_tensor(train_feats).float()  # [:, 0, :]
y_train = torch.as_tensor(train_labels[:, None]).long()
X_val = torch.as_tensor(valid_feats).float()  # [:, 0, :]
y_val = torch.as_tensor(valid_labels[:, None]).long()

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

class_counts = y_train.sum()
class_counts = torch.as_tensor([(class_counts) / len(y_train)])
class_weights = 1.0 / class_counts.float()
# Create DataLoader instances

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
# test_dataset = TensorDataset(X_test, y_test)

batch_size = 512  # You can adjust the batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the MLP
# input_size =   # Number of features in the input
# model = MLP(X_train.shape[-1], X_train.shape[-2]).cuda()
model = TFConv(X_train.shape[-1], X_train.shape[-2], mean, std).cuda()
ema = EMA(
    model,
    beta=0.9999,  # exponential moving average factor
    update_after_step=100,  # only after this number of .update() calls will it start updating
    update_every=1,
).cuda()
# 3. Training Loop
# criterion = FocalLoss()
criterion = nn.BCEWithLogitsLoss(weight=class_weights.cuda())

optimizer = optim.Adam(model.parameters(), lr=5e-4)  # Learning rate can be adjusted

num_epochs = 1000  # Number of epochs can be adjusted
val_freq = 1000
best_val_loss = 0
no_improv_thres = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    ct = 0
    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)[..., 0, :]  # Forward pass
        loss = criterion(outputs, labels.float())  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize
        ema.update()
        ct += 1
        if ct >= val_freq:
            ct = 0
            break

    # 4. Validation Step
    m = ema.ema_model
    val_loss, val_aupr = validate_model(m, val_loader, criterion, "cuda")
    print(f"Epoch: {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation AUPR: {val_aupr:.4f}")
    if val_aupr > best_val_loss:
        best_val_loss = val_aupr
        no_improv = 0
    else:
        no_improv += 1

    if no_improv >= no_improv_thres:
        break
    model.train()
    ema.train()
m2 = TFConv_translate(m)
print(m2)
m = m.eval()
m2 = m2.eval()
a, b = m(inputs), m2(inputs)
print(
    a.shape, b.shape, a.reshape((-1)), b.reshape((-1)), torch.allclose(a, b, atol=1e-3, rtol=1e-3)
)
from scipy.stats import pearsonr

print(pearsonr(a.reshape((-1)).cpu().detach().numpy(), b.reshape((-1)).cpu().detach().numpy()))
m2 = torch.jit.script(m2)
# Save to file
torch.jit.save(m2, f"TFBS_{feats[0]}_conv_v3.pt")
# torch.jit.save(m, f'TFBS_GM_{feats[0]}.pt')
