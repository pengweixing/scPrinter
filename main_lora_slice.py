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
torch.backends.cudnn.benchmark=True

import wandb
import random
import socket
import pickle
from wandb_main_lora_slice import main
from pathlib import Path

parser = argparse.ArgumentParser(description='Bindingscore BPNet')
parser.add_argument('--config', type=str, default='config.JSON', help='config file')
parser.add_argument('--data_dir', type=str, default='/', help='data directory, will be append to all data path in config')
parser.add_argument('--temp_dir', type=str, default='/', help='temp directory, will be used to store working models')
parser.add_argument('--model_dir', type=str, default='/', help='will be used to store final models')
parser.add_argument('--enable_wandb', action='store_true', help='enable wandb')
torch.set_num_threads(4)
args = parser.parse_args()
config = json.load(open(args.config))

for path in [args.data_dir, args.temp_dir, args.model_dir]:
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

config['data_dir'] = args.data_dir
config['temp_dir'] = args.temp_dir
config['model_dir'] = args.model_dir

if args.enable_wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="scPrinterSeq_v3_lora",
        notes=socket.gethostname() if "notes" not in config else config['notes'],
        # track hyperparameters and run metadata
        config=config,
        job_type="training",
        tags=config['tags'] if "tags" in config else [],
        reinit=True
    )


main(config, args.enable_wandb)