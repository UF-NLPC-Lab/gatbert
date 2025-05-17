# STL
import glob
import os
# 3rd Party
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import yaml
# Local
from .rgcn import CNTrainModule
from .data import CORPUS_PARSERS

if __name__ == "__main__":

    csv_path = "/home/ethanlmines/blue_dir/datasets/VAST/vast_dev.csv"
    corpus_type = 'vast'
    ckpt_path = "./graph_logs/version_1/checkpoints/epoch=0-step=25.ckpt"

    ckpt = torch.load(ckpt_path, weights_only=True)
    mod = CNTrainModule(**ckpt['hyper_parameters'])
    mod.load_state_dict(ckpt['state_dict'])

    parse_fn = CORPUS_PARSERS[corpus_type]
    sample_iter = parse_fn(csv_path)
    predict_dataloader = mod.make_predict_dataloader(sample_iter)

    trainer = L.Trainer()
    predictions = trainer.predict(mod, predict_dataloader)
