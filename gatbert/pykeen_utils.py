# STL
import os
# 3rd Party
import torch
import pykeen
import pykeen.datasets

def save_all_triples(ds: pykeen.datasets.Dataset, out_path: os.PathLike):
    concat_tripples = torch.concatenate([
        ds.training.mapped_triples,
        ds.validation.mapped_triples,
        ds.testing.mapped_triples
    ])
    concatenated_factory = ds.training.clone_and_exchange_triples(
        mapped_triples=concat_tripples
    )
    return concatenated_factory.to_path_binary(out_path)
