# STL
import argparse
import os
import pathlib
import importlib
import sys
# 3rd Party
import torch
import yaml
# Local
from .modules import *

def export(model_dir: os.PathLike, hf_repo: str):
    model_dir = pathlib.Path(model_dir)

    config_path = model_dir.joinpath("config.yaml")
    with open(config_path, 'r') as f:
        yaml_obj = yaml.safe_load(f)

    model_sec = yaml_obj['model']
    class_path = model_sec['class_path']
    init_args = model_sec['init_args']

    checkpoint_dir = model_dir.joinpath("checkpoints")
    checkpoint_paths = os.listdir(checkpoint_dir)
    assert len(checkpoint_paths) == 1, "We don't have support for loading one specific checkpoint out of many right now."
    checkpoint_path = checkpoint_dir.joinpath(checkpoint_paths[0])

    module_name, base_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, base_name)

    lightning_mod = cls(**init_args)

    # Any of our lightning mods that supports export will have a "wrapped" attribute
    hf_model = lightning_mod.wrapped
    # Same goes for the tokenizer attribute
    tokenizer = lightning_mod.tokenizer

    state_dict = torch.load(checkpoint_path, weights_only=True)
    lightning_mod.load_state_dict(state_dict['state_dict'])

    print("Attempting to upload model...")
    hf_model.push_to_hub(hf_repo)
    print("Attempting to upload tokenizer...")
    tokenizer.push_to_hub(hf_repo)

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="path/to/lightning/result_dir/")
    parser.add_argument("--hf", metavar="UF-NLPC-Lab/test_model", required=True, help="HF repo to upload model")
    args = parser.parse_args(raw_args)
    export(args.i, args.hf)

if __name__ == "__main__":
    main()
    model_dir = sys.argv[1]
    out_dir = sys.argv[2]