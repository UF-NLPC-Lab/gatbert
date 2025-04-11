# Local
import argparse
import sys
import os
# 3rd Party
from pykeen.datasets import ConceptNet
from pykeen.utils import set_random_seed
from pykeen.hpo import hpo_pipeline
# Local
from .constants import OPTUNA_STUDY_NAME
from .graph import GraphPaths


def main(raw_args):
    parser = argparse.ArgumentParser(description="Generate graph embeddings")

    parser.add_argument("--graph", metavar="graph/", required=True,
                        help="Path to dir with conceptnet assertions, and where to write output")
    parser.add_argument("--timeout", type=int, default=3600, metavar="3600")
    args = parser.parse_args(raw_args)

    graph_paths = GraphPaths(args.graph)
    os.makedirs(args.graph, exist_ok=True)

    set_random_seed(0)
    create_inverse_triples = True
    
    ds = ConceptNet(name=graph_paths.assertions_path, create_inverse_triples=create_inverse_triples)

    hpo_res = hpo_pipeline(
        timeout=args.timeout,
        dataset=ds,
        model='TransE',
        model_kwargs_ranges=dict(
            embedding_dim=dict(type=int, low=64, high=1024, step=64)
        ),
        training_kwargs=dict(
            batch_size=128,
        ),
        stopper='early',
        stopper_kwargs=dict(
            frequency=5,
            patience=2,
        ),
        study_name=OPTUNA_STUDY_NAME,
        storage=f"sqlite:///{graph_paths.optuna_db}",
        load_if_exists=True,
    )

    hpo_res.save_to_directory(args.graph)


if __name__ == "__main__":
    main(raw_args=sys.argv[1:])