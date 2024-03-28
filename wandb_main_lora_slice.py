import os.path

from scprinter.seq.dataloader import *
from scprinter.seq.scdataloader import *
from scprinter.seq.Modules import *
from scprinter.seq.Models import  *
from scprinter.seq.minimum_footprint import *
from scprinter.seq.ema import EMA
from scprinter.utils import load_entire_hdf5, loadDispModel

import argparse
import json
import pandas as pd
import torch
import gc
from copy import deepcopy
import h5py
import time
import numpy as np

import transformers
import scprinter as scp

import wandb
import random
import socket
import pickle
from wandb_main import construct_model_from_config

from sklearn.preprocessing import StandardScaler

def main(config=None, enable_wandb=True):
    # Initialize a new wandb run
    with (((wandb.init(config=config)))):
        print ("start a new run!!!")
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        disp_path = scp.datasets.pretrained_dispersion_model
        dispmodel = loadDispModel(disp_path)
        dispmodel = dispModel(deepcopy(dispmodel)).cuda()

        print("start a new run!!!")
        print("run id", wandb.run.name)

        config = wandb.config

        data_dir = config['data_dir']
        temp_dir = config['temp_dir']
        model_dir = config['model_dir']

        split = config['split']
        peaks = os.path.join(data_dir,config['peaks'])

        insertion = os.path.join(data_dir,config['insertion'])
        grp2barcodes = os.path.join(data_dir,config['grp2barcodes'])
        grp2embeddings = os.path.join(data_dir,config['grp2embeddings'])
        cells_of_interest = np.array(config['cells'])

        pretrain_model = os.path.join(data_dir,config['pretrain_model'])
        # coverages = os.path.join(data_dir, config['coverages'])




        peaks = pd.read_table(peaks, sep='\t', header=None)
        summits = peaks
        summits['summits'] = (summits[1] + summits[2]) // 2

        summits = summits[[0, 'summits']]
        summits['index'] = np.arange(len(summits))
        print(summits)
        modes = np.arange(2, 101, 1)
        insertion = pickle.load(open(insertion, 'rb'))
        grp2barcodes = np.array(pickle.load(open(grp2barcodes, 'rb')))[cells_of_interest]
        grp2embeddings = np.array(pickle.load(open(grp2embeddings, 'rb')))[cells_of_interest]

        max_jitter = config['max_jitter']
        ema_flag = config['ema']
        amp_flag = config['amp']
        batch_size = config['batch_size']
        weight_decay = config['weight_decay']
        genome = config['genome']
        cell_sample = config['cell_sample']
        if genome == 'hg38':
            genome = scp.genome.hg38
        elif genome == 'mm10':
            genome = scp.genome.mm10
        else:
            raise ValueError("genome not supported")

        lr = config['lr']
        method = config['method']
        if method == 'lora':
            acc_model = torch.load(pretrain_model)
            for p in acc_model.parameters():
                p.requires_grad = False
        else:
            raise ValueError("method not supported")

        total_params = sum(p.numel()
                           for p in acc_model.parameters())
        print("total params - pretrained model", total_params)


        print("output len", acc_model.output_len)
        print("dna len", acc_model.dna_len)

        # if os.path.exists(coverages):
        #     cov = pickle.load(open(coverages, 'rb'))
        #     save_coverage = False
        # else:
        cov = {k:None for k in split}
        save_coverage = False

        datasets = {k: scChromBPDataset(
            insertion_dict=insertion,
            bias = str(genome.fetch_bias())[:-3] + ".bw",
            group2idx=grp2barcodes,
            ref_seq=genome.fetch_fa(),
            summits=summits[summits[0].isin(split[k])],
            DNA_window=acc_model.dna_len,
            signal_window=acc_model.output_len + 200,
            max_jitter=max_jitter if k in ['train'] else 0,
            cached=True,
            lazy_cache=True,
            reverse_compliment=config['reverse_compliment'] if k in ['train'] else False,
            device='cpu',
            coverages=cov[k],
            data_augmentation=True if k in ['train'] else False,
            mode=config['dataloader_mode'] if k in ['train'] else 'uniform',
            cell_sample=cell_sample
        ) for k in split}

        # if save_coverage:
        #     cov = {k: datasets[k].coverages for k in split}
        #     pickle.dump(cov, open(coverages, 'wb'))

        dataloader = {
            k: ChromBPDataLoader(
                dataset=datasets[k],
                batch_size= (batch_size // cell_sample) * 2 if (config['dataloader_mode'] == 'peak') and (k in ['train'])  else batch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=True,
                collate_fn=collate_fn_singlecell if (config['dataloader_mode'] == 'peak') and (k in ['train']) else None) for k in split}

        acc_model = acc_model.cpu()
        acc_model = scFootprintBPNet(
            dna_cnn_model=acc_model.dna_cnn_model,
            hidden_layer_model=acc_model.hidden_layer_model,
            profile_cnn_model=acc_model.profile_cnn_model,
            dna_len=acc_model.dna_len,
            output_len=acc_model.output_len,
            embeddings=grp2embeddings,
            rank=config['lora_rank'],
            hidden_dim=config['lora_hidden_dim'],
            lora_dna_cnn=config['lora_dna_cnn'],
            lora_dilated_cnn=config['lora_dilated_cnn'],
            lora_pff_cnn=config['lora_pff_cnn'],
            lora_output_cnn=config['lora_output_cnn'],
            lora_count_cnn = config['lora_count_cnn'],
            method=method,
            n_lora_layers=config['n_lora_layers'],
        )

        pretrain_lora_model = os.path.join(data_dir,config['pretrain_lora_model'])
        pretrain_lora_model = torch.load(pretrain_lora_model)


        acc_model.profile_cnn_model.conv_layer.layer.load_state_dict(pretrain_lora_model.profile_cnn_model.conv_layer.layer.state_dict())
        acc_model.profile_cnn_model.linear.layer.load_state_dict(pretrain_lora_model.profile_cnn_model.linear.layer.state_dict())

        for flag, module, pretrain_module in [(config['lora_dna_cnn'], [acc_model.dna_cnn_model.conv], [pretrain_lora_model.dna_cnn_model.conv]),
                             (config['lora_dilated_cnn'],[l.module.conv1 for l in acc_model.hidden_layer_model.layers],
                              [l.module.conv1 for l in pretrain_lora_model.hidden_layer_model.layers]),
                             (config['lora_pff_cnn'], [l.module.conv2 for l in acc_model.hidden_layer_model.layers],
                              [l.module.conv2 for l in pretrain_lora_model.hidden_layer_model.layers]),
                             (config['lora_output_cnn'], [acc_model.profile_cnn_model.conv_layer], [pretrain_lora_model.profile_cnn_model.conv_layer]),
                             (config['lora_count_cnn'], [acc_model.profile_cnn_model.linear], [pretrain_lora_model.profile_cnn_model.linear])]:
            if flag:
                for m,prem in zip(module, pretrain_module):
                    for i in range(len(m.A_embedding)):
                        if not isinstance(m.A_embedding[i], nn.Embedding):
                            m.A_embedding[i].load_state_dict(prem.A_embedding[i].state_dict())
                    for i in range(len(m.B_embedding)):
                        if not isinstance(m.B_embedding[i], nn.Embedding):
                            m.B_embedding[i].load_state_dict(prem.B_embedding[i].state_dict())



        acc_model.cuda()

        print("model")
        print(acc_model)
        total_params = sum(p.numel()
                           for p in acc_model.parameters() if p.requires_grad)
        print("total trainable params", total_params)
        total_params = sum(p.numel()
                           for p in acc_model.parameters())
        print("total params", total_params)


        ema = None
        if ema_flag:
            update_after_step = 100
            print ("update after step", update_after_step)
            ema = EMA(
                acc_model,
                beta=0.9999,  # exponential moving average factor
                update_after_step=update_after_step,  # only after this number of .update() calls will it start updating
                update_every=10)  # how often to actually update, to save on compute (updates every 10th .update() call)

        torch.cuda.empty_cache()
        optimizer = torch.optim.AdamW(acc_model.parameters(), lr=lr, weight_decay=weight_decay)
        if "scheduler" in config:
            scheduler = config['scheduler']
        else:
            scheduler = False
        if scheduler:
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=3000,
                                                                     num_training_steps=100000)
        else:
            scheduler = None

        val_loss, profile_pearson, across_pearson = validation_step_footprint(acc_model,
                                                                              dataloader['valid'],
                                                                              (len(
                                                                                datasets['valid'].summits) // batch_size),
                                                                              dispmodel,
                                                                              modes)
        print ("before lora", val_loss, profile_pearson, across_pearson)
        acc_model.fit(dispmodel,
                      dataloader['train'],
                      validation_data=dataloader['valid'],
                      validation_size=(len(datasets['valid'].summits) // batch_size),
                      max_epochs=300,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      validation_freq=(len(datasets['train'].summits) // batch_size),
                      early_stopping=5 if 'early_stopping' not in config else config[
                          'early_stopping'],
                      return_best=True,
                      savename=f'{temp_dir}/{wandb.run.name}',
                      modes=modes,
                      downsample=None if 'downsample' not in config else config[
                          'downsample'],
                      ema=ema,
                      use_amp=amp_flag,
                      use_wandb=enable_wandb,
                      accumulate_grad=config["accumulate_grad_batches"],
                      batch_size=batch_size,)
        if ema:
            del acc_model
            acc_model = torch.load(f'{temp_dir}/{wandb.run.name}.ema_model.pt').cuda()
        acc_model.eval()
        savename = config['savename']
        torch.save(acc_model, f'{model_dir}/{savename}-{wandb.run.name}.pt')
        valid_loss, valid_within, valid_across = validation_step_footprint(acc_model,
                                                                           dataloader['valid'],
                                                                           (len(
                                                                               datasets['valid'].summits) // batch_size),
                                                                           dispmodel,
                                                                           modes)
        valid_loss, valid_within, valid_across = float(valid_loss[0]), float(valid_within[0]), float(
            valid_across[0])
        test_loss, test_within, test_across = validation_step_footprint(acc_model,
                                                                        dataloader['test'],
                                                                        (len(
                                                                            datasets['valid'].summits) // batch_size),
                                                                        dispmodel,
                                                                        modes)
        test_loss, test_within, test_across = float(test_loss[0]), float(test_within[0]), float(test_across[0])
        if enable_wandb:
            wandb.summary['final_valid_loss'] = valid_loss
            wandb.summary['final_valid_within'] = valid_within
            wandb.summary['final_valid_across'] = valid_across
            wandb.summary['final_test_loss'] = test_loss
            wandb.summary['final_test_within'] = test_within
            wandb.summary['final_test_across'] = test_across
            wandb.finish()
        del acc_model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
