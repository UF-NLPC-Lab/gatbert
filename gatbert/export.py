# STL
import os
import pathlib
import importlib
import sys
# 3rd Party
import torch
import yaml
# Local
from .stance_classifier import BertClassifier
from .modules import MyStanceModule

def export(model_dir: os.PathLike, out_dir: os.PathLike):
    model_dir = pathlib.Path(model_dir)
    out_dir = pathlib.Path(out_dir)

    config_path = model_dir.joinpath("config.yaml")
    with open(config_path, 'r') as f:
        yaml_obj = yaml.safe_load(f)

    classifier_sec = yaml_obj['model']['classifier']
    class_path = classifier_sec['class_path']
    init_args = classifier_sec['init_args']

    checkpoint_dir = model_dir.joinpath("checkpoints")
    checkpoint_paths = os.listdir(checkpoint_dir)
    assert len(checkpoint_paths) == 1
    checkpoint_path = checkpoint_dir.joinpath(checkpoint_paths[0])

    module_name, base_name = class_path.rsplit('.', 1)
    # Assume module already imported. I don't like dynamic imports.
    module = importlib.import_module(module_name)
    cls = getattr(module, base_name)

    classifier = cls(**init_args)
    module = MyStanceModule(classifier)
    state_dict = torch.load(checkpoint_path, weights_only=True)
    module.load_state_dict(state_dict['state_dict'])

    if isinstance(classifier, BertClassifier):
        bert = classifier.bert
        tokenizer = classifier.tokenizer
        bert.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
    else:
        raise ValueError("Only BertClassifier supported")

if __name__ == "__main__":
    model_dir = sys.argv[1]
    out_dir = sys.argv[2]
    export(model_dir, out_dir)