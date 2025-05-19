# 3rd Party
import numpy as np
import torch
import lightning as L
# Local
from .rgcn import CNTrainModule
from .data import CORPUS_PARSERS

if __name__ == "__main__":

    csv_path = "/home/ethanlmines/blue_dir/datasets/VAST/vast_dev.csv"
    corpus_type = 'vast'
    ckpt_path = "./graph_logs/version_1/checkpoints/epoch=0-step=25.ckpt"
    out_path = "./temp/vast_dev_graph.npy"

    ckpt = torch.load(ckpt_path, weights_only=True)
    mod = CNTrainModule(**ckpt['hyper_parameters'])
    mod.load_state_dict(ckpt['state_dict'])

    parse_fn = CORPUS_PARSERS[corpus_type]
    sample_iter = parse_fn(csv_path)
    predict_dataloader = mod.make_predict_dataloader(sample_iter)

    trainer = L.Trainer()
    predictions = trainer.predict(mod, predict_dataloader)
    predictions = [pred.numpy() for pred in predictions]
    predictions = np.stack(predictions)
    np.save(out_path, predictions, allow_pickle=False)