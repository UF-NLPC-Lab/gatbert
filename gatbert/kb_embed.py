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
from pykeen.models import ConvE, DistMult, ProjE, PairRE, TransE, TransF, TransH, TransR
from pykeen.utils import set_random_seed
# Local
from .pykeen_utils import save_all_triples
from .graph import get_entity_embeddings, get_relation_embeddings

SIMPLE_ENTITY_EMBEDDINGS = (ConvE, DistMult, PairRE, ProjE, TransE, TransF, TransH, TransR,)
SIMPLE_REL_EMBEDDINGS = (TransE,)

def main(raw_args):
    parser = argparse.ArgumentParser(description="Generate graph embeddings")

    parser.add_argument("-cn", metavar="conceptnet-assertions-5.7.0.csv", required=True,
                        help="Path to conceptnet assertions")
    parser.add_argument("--embed", metavar='TransE',
                        help="Embedding type. Don't specify to just get all the relation triples")
    parser.add_argument("--dim", metavar="50", type=int, default=50, help="Embedding dimensionality")
    parser.add_argument("--epochs", metavar="5", type=int, default=5, help="Embedding dimensionality")

    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--freq", metavar="10", type=int, default=10, help="Embedding dimensionality")
    parser.add_argument("--patience", metavar="2", type=int, default=2, help="Embedding dimensionality")

    parser.add_argument("-o", metavar="output_dir/",
                        help="Output directory containing all the triples, and a torch.nn.Embedding save model")
    parser.add_argument("--seed", type=int, default=1, metavar="1", help="Random seed for pykeen")
    args = parser.parse_args(raw_args)

    assert args.embed is None or any(args.embed in cls.__name__ for cls in SIMPLE_ENTITY_EMBEDDINGS)

    set_random_seed(args.seed)
    create_inverse_triples = True
    
    ds = ConceptNet(name=args.cn, create_inverse_triples=create_inverse_triples)
    os.makedirs(args.o, exist_ok=True)
    out_path = pathlib.Path(args.o)
    if args.embed:
        kwargs = {}
        if args.early_stopping:
            kwargs['stopper'] = "early"
            kwargs['stopper_kwargs'] = {"frequency": args.freq, "patience": args.patience}
        pipeline_res = pipeline(
            dataset=ds,
            model=args.embed,
            model_kwargs={"embedding_dim": args.dim},
            random_seed=args.seed,
            training_kwargs={"num_epochs": args.epochs},
            **kwargs
        )
        model = pipeline_res.model
        if isinstance(model, SIMPLE_ENTITY_EMBEDDINGS):
            torch.save(model.entity_representations[0]._embeddings, get_entity_embeddings(out_path))
        else:
            raise ValueError(f"Unsupported type {type(model)}")
        if isinstance(model, SIMPLE_REL_EMBEDDINGS):
            torch.save(model.relation_representations[0]._embeddings, get_relation_embeddings(out_path))

        pipeline_res.save_to_directory(
            out_path,
            # The "replicates" is the model file. Already doing that manually above
            save_replicates=False, 
            # Also already saving the training triples (along with the val and test ones) manually below
            save_training=False,
        )
    save_all_triples(ds, out_path)



if __name__ == "__main__":
    main(raw_args=sys.argv[1:])