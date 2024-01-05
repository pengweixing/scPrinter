from .dataloader import *
from .Modules import *
from .Models import  *
from .minimum_footprint import *
from .ema import EMA

from ..utils import loadDispModel, load_entire_hdf5

import argparse
import json
import pandas as pd
import torch
import gc
from copy import deepcopy
import h5py
import time
import numpy as np
import scprinter as scp
import random
import socket


def train_model(model_configs,
                device='cuda',
                wandb_info=None):

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    if wandb_info is not None:
        import wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_info['project'],
            notes=socket.gethostname(),
            # track hyperparameters and run metadata
            config=model_configs,
            job_type="training",
            reinit=True
        )

    torch.set_num_threads(4)
    disp_path = scp.datasets.pretrained_dispersion_model
    dispmodel = loadDispModel(disp_path)
    dispmodel = dispModel(deepcopy(dispmodel)).cuda()

    # Load everything from the config file
    config = model_configs
    split = config['split']
    signals = config['signals']
    bias = config['bias']
    peaks = config['peaks']
    reverse_compliment = config['reverse_compliment']
    max_jitter = config['max_jitter']
    n_filters = config['n_filters']
    n_layers = config['n_layers']
    activation = config['activation']
    head_kernel_size = config['head_kernel_size']
    kernel_size = config['kernel_size']
    batch_norm = config['batch_norm']
    dilation_base = config['dilation_base']
    bottleneck_factor = config['bottleneck_factor']
    rezero = config['rezero']
    batch_norm_momentum = config['batch_norm_momentum']
    groups = config['groups']
    no_inception = config['no_inception']
    n_inception_layers = config['n_inception_layers']
    inception_layers_after = config['inception_layers_after']
    inception_version = config['inception_version']
    ema_flag = config['ema']
    amp_flag = config['amp']
    batch_size = config['batch_size']
    weight_decay = config['weight_decay']
    lr = config['lr']
    savename = config['savename']

    peaks = pd.read_table(peaks, sep='\t', header=None)
    modes = np.arange(2, 101, 1)
    if activation == 'relu':
        activation = torch.nn.ReLU()
    elif activation == 'gelu':
        activation = torch.nn.GELU()

    bottleneck = int(n_filters * bottleneck_factor)

    if no_inception:
        n_inception_layers = 0

    if inception_layers_after:
        inception_bool = [False] * (n_layers - n_inception_layers) + [True] * (n_inception_layers)
    else:
        inception_bool = [True] * n_inception_layers + [False] * (n_layers - n_inception_layers)

    acc_dna_cnn = DNA_CNN(n_filters=n_filters, )
    dilation_func = lambda x: 2 ** (x + dilation_base)
    acc_hidden = DilatedCNN(n_filters=n_filters,
                            bottleneck=bottleneck,
                            n_layers=n_layers,
                            kernel_size=kernel_size,
                            groups=groups,
                            activation=activation,
                            batch_norm=batch_norm,
                            residual=True,
                            rezero=rezero,
                            dilation_func=dilation_func,
                            batch_norm_momentum=batch_norm_momentum,
                            inception=inception_bool,
                            inception_version=inception_version
                            )

    acc_head = Footprints_head(n_filters,
                               kernel_size=head_kernel_size,
                               n_scales=len(modes),
                               per_peak_feats=1)
    output_len = 800
    dna_len = output_len + acc_dna_cnn.conv.weight.shape[2] - 1
    for i in range(n_layers):
        dna_len = dna_len + 2 * (kernel_size // 2) * dilation_func(i)


    acc_model = FootprintBPNet(
        dna_cnn_model=acc_dna_cnn,
        hidden_layer_model=acc_hidden,
        profile_cnn_model=acc_head,
        dna_len=dna_len,
        output_len=output_len)

    acc_model = acc_model.to(device)

    ema = None
    if ema_flag:
        ema = EMA(
            acc_model,
            beta=0.9999,  # exponential moving average factor
            update_after_step=100,  # only after this number of .update() calls will it start updating
            update_every=10)  # how often to actually update, to save on compute (updates every 10th .update() call)

    print("model")
    print(acc_model)
    total_params = sum(p.numel() for p in acc_model.parameters() if p.requires_grad)
    print("dna_len", dna_len)
    print("output len", output_len)
    print("total params", total_params)
    acc_model.dna_len = dna_len
    acc_model.signal_len = output_len

    summits = peaks
    summits = summits.drop_duplicates([0, 1, 2])  # drop exact same loci
    summits['summits'] = summits[1] - 1 + 500
    summits = summits[[0, 'summits']]
    summits['index'] = np.arange(len(summits))
    signals = [signals[0], bias]
    datasets = {k: ChromBPDataset(
        signals=signals,
        ref_seq=scp.genome.hg38.fetch_fa(),
        summits=summits[summits[0].isin(split[k])],
        DNA_window=dna_len,
        signal_window=output_len + 200,
        max_jitter=max_jitter if k in ['train'] else 0,
        min_counts=None,
        max_counts=None,
        cached=False,
        lazy_cache=True,
        reverse_compliment=reverse_compliment if k in ['train'] else False,
        device='cpu'
    ) for k in split}

    coverage = datasets['train'].coverage

    min_, max_ = np.quantile(coverage, 0.0001), np.quantile(coverage, 0.9999)
    print("coverage cutoff", min_, max_)
    for k in split:
        datasets[k].filter_by_coverage(min_, max_)

    dataloader = {
        k: ChromBPDataLoader(
            dataset=datasets[k],
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=True if k in ['train'] else False) for k in split}

    torch.cuda.empty_cache()



    optimizer = torch.optim.AdamW(acc_model.parameters(), lr=lr, weight_decay=weight_decay)
    n_epoch, loss_history = acc_model.fit(dispmodel,
                                          dataloader['train'],
                                          validation_data=dataloader['valid'],
                                          validation_size=None,
                                          max_epochs=300,
                                          optimizer=optimizer,
                                          scheduler=None,
                                          validation_freq=None,
                                          early_stopping=5 if 'early_stopping' not in config else config[
                                              'early_stopping'],
                                          return_best=True,
                                          savename=savename,
                                          modes=modes,
                                          coverage_weight=0.1,
                                          downsample=None if 'downsample' not in config else config[
                                              'downsample'],
                                          ema=ema,
                                          use_amp=amp_flag,
                                          use_wandb=wandb_info is not None, )
    torch.save(acc_model, savename + '.model.pt')
    if ema:
        del acc_model
        acc_model = torch.load(savename + '.ema_model.pt').cuda()
    valid_loss, valid_within, valid_across = validation_step_footprint(acc_model,
                                                                       dataloader['valid'],
                                                                       None,
                                                                       dispmodel,
                                                                       modes, verbose=True)
    valid_loss, valid_within, valid_across = float(valid_loss[0]), float(valid_within[0]), float(
        valid_across[0])
    test_loss, test_within, test_across = validation_step_footprint(acc_model,
                                                                    dataloader['test'],
                                                                    None,
                                                                    dispmodel,
                                                                    modes, verbose=True)
    test_loss, test_within, test_across = float(test_loss[0]), float(test_within[0]), float(test_across[0])
    if wandb_info is not None:
        wandb.summary['final_valid_loss'] = valid_loss
        wandb.summary['final_valid_within'] = valid_within
        wandb.summary['final_valid_across'] = valid_across
        wandb.summary['final_test_loss'] = test_loss
        wandb.summary['final_test_within'] = test_within
        wandb.summary['final_test_across'] = test_across
        wandb.finish()
    gc.collect()
    torch.cuda.empty_cache()
    return acc_model