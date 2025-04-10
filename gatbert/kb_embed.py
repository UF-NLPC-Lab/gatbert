# Local
import argparse
import sys
import os
# 3rd Party
import torch
from pykeen.datasets import ConceptNet
from pykeen.pipeline import pipeline
from pykeen.utils import set_random_seed
# Local
from .pykeen_utils import save_all_triples
from .graph import GraphPaths


def main(raw_args):
    parser = argparse.ArgumentParser(description="Generate graph embeddings")

    parser.add_argument("--graph", metavar="graph/", required=True,
                        help="Path to dir with conceptnet assertions, and where to write output")

    parser.add_argument("--embed", metavar='TransE',
                        help="Embedding type. Don't specify to just get all the relation triples")
    parser.add_argument("--dim", metavar="64", type=int, default=64, help="Embedding dimensionality")
    parser.add_argument("--epochs", metavar="5000", type=int, default=5000, help="Embedding dimensionality")

    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--freq", metavar="10", type=int, default=10, help="Validation frequency (# epochs)")
    parser.add_argument("--patience", metavar="2", type=int, default=2, help="# Maximum allowed validation checks without improvement in metrics")

    parser.add_argument("--seed", type=int, default=1, metavar="1", help="Random seed for pykeen")
    args = parser.parse_args(raw_args)

    assert args.embed is None or args.embed == 'TransE', "Only TransE supported currently"
    graph_paths = GraphPaths(args.graph)
    os.makedirs(args.graph, exist_ok=True)

    set_random_seed(args.seed)
    create_inverse_triples = True
    
    ds = ConceptNet(name=graph_paths.assertions_path, create_inverse_triples=create_inverse_triples)
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
            training_kwargs={
                "num_epochs": args.epochs,

                "checkpoint_directory": graph_paths.checkpoint_dir,
                "checkpoint_frequency": 30,
                "checkpoint_name": "checkpoint.pt"
            },
            **kwargs
        )
        model = pipeline_res.model
        torch.save(model.entity_representations[0]._embeddings, graph_paths.entity_embeddings_path)
        torch.save(model.relation_representations[0]._embeddings, graph_paths.relation_embeddings_path)
        pipeline_res.save_to_directory(
            args.graph,
            # The "replicates" is the model file. Already doing that manually above
            save_replicates=False, 
            # Also already saving the training triples (along with the val and test ones) manually below
            save_training=False,
        )
    save_all_triples(ds, args.graph)



if __name__ == "__main__":
    main(raw_args=sys.argv[1:])