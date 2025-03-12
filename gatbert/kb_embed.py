# Local
import argparse
import sys
import os
import pathlib
from typing import Tuple
# 3rd Party
import torch
from pykeen.datasets import ConceptNet
from pykeen.pipeline import pipeline
from pykeen.models import TransE
from pykeen.utils import set_random_seed
# Local
from .graph import get_entity_embeddings, get_relation_embeddings

def extract_transe(model: TransE) -> Tuple[torch.nn.Embedding, torch.nn.Embedding]:
    return model.entity_representations[0]._embeddings, model.relation_representations[0]._embeddings

def extract_embeddings(model) -> Tuple[torch.nn.Embedding, torch.nn.Embedding]:
    if isinstance(model, TransE):
        return extract_transe(model)
    raise ValueError(f"Unsupported type {type(model)}")

def main(raw_args):
    parser = argparse.ArgumentParser(description="Generate graph embeddings")

    parser.add_argument("-cn", metavar="conceptnet-assertions-5.7.0.csv", required=True,
                        help="Path to conceptnet assertions")
    parser.add_argument("--embed", metavar='TransE',
                        help="Embedding type. Don't specify to just get all the relation triples")
    parser.add_argument("--dim", metavar="50", type=int, default=50, help="Embedding dimensionality")
    parser.add_argument("-o", metavar="output_dir/",
                        help="Output directory containing all the triples, and a torch.nn.Embedding save model")
    parser.add_argument("--seed", type=int, default=1, metavar="1", help="Random seed for pykeen")
    args = parser.parse_args(raw_args)

    assert args.embed is None or args.embed in {'TransE'}

    set_random_seed(args.seed)
    create_inverse_triples = True
    
    ds = ConceptNet(name=args.cn, create_inverse_triples=create_inverse_triples)
    os.makedirs(args.o, exist_ok=True)
    out_path = pathlib.Path(args.o)
    if args.embed:
        pipeline_res = pipeline(
            dataset=ds,
            model=args.embed,
            model_kwargs={"embedding_dim": args.dim},
            random_seed=args.seed,
        )
        entity_embeddings, relation_embeddings = extract_embeddings(pipeline_res.model)
        torch.save(entity_embeddings, get_entity_embeddings(out_path))
        torch.save(relation_embeddings, get_relation_embeddings(out_path))

        pipeline_res.save_to_directory(
            out_path,
            # The "replicates" is the model file. Already doing that manually above
            save_replicates=False, 
            # Also already saving the training triples (along with the val and test ones) manually below
            save_training=False
        )

    concat_tripples = torch.concatenate([
        ds.training.mapped_triples,
        ds.validation.mapped_triples,
        ds.testing.mapped_triples
    ])
    concatenated_factory = ds.training.clone_and_exchange_triples(
        mapped_triples=concat_tripples
    )
    concatenated_factory.to_path_binary(out_path)



if __name__ == "__main__":
    main(raw_args=sys.argv[1:])