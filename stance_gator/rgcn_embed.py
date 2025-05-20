# STL
import argparse
# 3rd Party
import numpy as np
import torch
import lightning as L
# Local
from .rgcn import CNEncoder
from .data import add_corpus_args, get_corpus_parser

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    add_corpus_args(parser)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument('-o', metavar="embeddings.npy", required=True)
    args = parser.parse_args(raw_args)

    sample_iter = get_corpus_parser(args)

    ckpt = torch.load(args.ckpt, weights_only=True)
    mod = CNEncoder(**ckpt['hyper_parameters'])
    mod.load_state_dict(ckpt['state_dict'])

    predict_dataloader = mod.make_predict_dataloader(sample_iter)
    trainer = L.Trainer(deterministic=True, logger=False)
    predictions = trainer.predict(mod, predict_dataloader)
    predictions = [pred.numpy() for pred in predictions]
    predictions = np.stack(predictions)
    np.save(args.o, predictions, allow_pickle=False)

if __name__ == "__main__":
    main()