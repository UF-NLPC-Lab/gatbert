# STL
import os
# 3rd Party
import torch
import pykeen
import pykeen.datasets
from pykeen.triples import CoreTriplesFactory

def get_all_triples(ds: pykeen.datasets.Dataset) -> CoreTriplesFactory:
    concat_tripples = torch.concatenate([
        ds.training.mapped_triples,
        ds.validation.mapped_triples,
        ds.testing.mapped_triples
    ])
    return ds.training.clone_and_exchange_triples(
        mapped_triples=concat_tripples
    )

def save_all_triples(ds: pykeen.datasets.Dataset, out_path: os.PathLike):
    return get_all_triples(ds).to_path_binary(out_path)
