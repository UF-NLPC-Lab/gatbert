# STL
from typing import Optional
import argparse
import os
import pathlib
import sys
import gzip
# 3rd Party
import torch
from tqdm import tqdm
# Local
from .constants import SpecialRelation
from .utils import batched
from .graph import get_entity_embeddings_path, get_bert_triples_path

def make_edges(graph_root: os.PathLike,
         out_path: Optional[os.PathLike] = None,
         threshold: float = 0.85):
    graph_root = pathlib.Path(graph_root)
    if out_path is None:
        out_path = get_bert_triples_path(graph_root)

    assert os.path.exists(graph_root)
    embedding_mat = torch.load(get_entity_embeddings_path(graph_root), weights_only=False).weight
    with torch.no_grad():
        for i in tqdm(range(embedding_mat.shape[0]), total=embedding_mat.shape[0]):
            embedding_mat[i] /= torch.linalg.vector_norm(embedding_mat[i])
    batch_size = 32
    est_batches = embedding_mat.shape[0] // batch_size
    rel_id = str(SpecialRelation.KB_SIM.value)
    # Afterward need to filter out the self-loops
    with torch.no_grad(), gzip.open(out_path, 'wb') as w:
        w.write(('\t'.join(["head", "relation", "tail", "similarity"]) + '\n').encode())

        for tail_inds in tqdm(batched(range(embedding_mat.shape[0]), batch_size), total=est_batches):
            tail_sim_vals = embedding_mat @ embedding_mat[tail_inds].transpose(1, 0)
            meet_threshold = torch.where(tail_sim_vals > threshold)
            tup_iter = zip(meet_threshold[0].tolist(),
                           (meet_threshold[1] + tail_inds[0]).tolist(),
                           tail_sim_vals[meet_threshold].tolist())
            tup_iter = filter(lambda p: p[0] != p[1], tup_iter) # Ignore self-loops
            w.write((
                "\n".join([
                    "\t".join([str(h), rel_id, str(p), f"{sim:.04f}"]) for h,p,sim in tup_iter
                ]) + '\n'
            ).encode())

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.85, metavar="0.85")
    parser.add_argument("-d", type=pathlib.Path, metavar="graph_dir/", required=True)
    parser.add_argument("-o", type=pathlib.Path, metavar="bert_triples.tsv.gz")
    args = parser.parse_args(raw_args)
    make_edges(args.d, args.o, threshold=args.threshold)

if __name__ == "__main__":
    main(sys.argv[1:])