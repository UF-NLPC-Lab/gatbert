# STL
import sys
import os
from typing import Dict, Any
import importlib
# 3rd Party
import lightning as L
import torch

def load_module(ckpt_path: os.PathLike) -> L.LightningModule:
    ckpt = torch.load(ckpt_path)
    hparams = ckpt['hyper_parameters']
    _class_path = hparams.pop('_class_path')
    last_dot = _class_path.rfind('.')
    module_name = _class_path[:last_dot]
    class_name = _class_path[last_dot + 1:]
    module_name, class_name
    code_module = importlib.import_module(module_name)

    cls = getattr(code_module, class_name)
    hparams.pop('_instantiator')
    l_mod = cls(**hparams)
    l_mod.load_state_dict(ckpt['state_dict'])
    return l_mod