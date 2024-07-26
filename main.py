import argparse
import json

from scprinter.seq.Modules import *

torch.backends.cudnn.benchmark = True

import socket
from pathlib import Path

import wandb

from wandb_main import main

parser = argparse.ArgumentParser(description="Bindingscore BPNet")
parser.add_argument("--config", type=str, default="config.JSON", help="config file")
parser.add_argument(
    "--data_dir",
    type=str,
    default="/",
    help="data directory, will be append to all data path in config",
)
parser.add_argument(
    "--temp_dir",
    type=str,
    default="/",
    help="temp directory, will be used to store working models",
)
parser.add_argument("--model_dir", type=str, default="/", help="will be used to store final models")
parser.add_argument("--enable_wandb", action="store_true", help="enable wandb")
parser.add_argument("--shap", action="store_true", help="enable shap")
parser.add_argument("--project", type=str, default="scPrinterSeq_v3", help="project name")
torch.set_num_threads(4)
args = parser.parse_args()
config = json.load(open(args.config))


for path in [args.data_dir, args.temp_dir, args.model_dir]:
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

config["data_dir"] = str(args.data_dir)
config["temp_dir"] = str(args.temp_dir)
config["model_dir"] = str(args.model_dir)
config["shap"] = args.shap

if args.enable_wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=args.project,
        notes=socket.gethostname() if "notes" not in config else config["notes"],
        tags=config["tags"] if "tags" in config else [],
        # track hyperparameters and run metadata
        config=config,
        job_type="training",
        reinit=True,
    )
main(config, args.enable_wandb)
