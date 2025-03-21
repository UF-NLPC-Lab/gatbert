"""
Usage: python -m gatbert.bert_sim graph_dir/

graph_dir/ should contain an `entity_to_id.tsv.gz`.
A `numeric_bert_triples.tsv` is written as output.
"""
# STL
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
from .graph import get_entity_embeddings, get_bert_triples_path

def main(graph_root, threshold: float = 0.95):
    graph_root = pathlib.Path(graph_root)
    assert os.path.exists(graph_root)
    out_path = get_bert_triples_path(graph_root)
    embedding_mat = torch.load(get_entity_embeddings(graph_root), weights_only=False).weight
    with torch.no_grad():
        for i in tqdm(range(embedding_mat.shape[0]), total=embedding_mat.shape[0]):
            embedding_mat[i] /= torch.linalg.vector_norm(embedding_mat[i])
    batch_size = 32
    est_batches = embedding_mat.shape[0] // batch_size
    rel_id = str(SpecialRelation.KB_SIM.value)
    # Afterward need to filter out the self-loops
    with torch.no_grad(), gzip.open(out_path, 'wb') as w:
        w.write(('\t'.join(["head", "relation", "tail"]) + '\n').encode())

        for tail_inds in tqdm(batched(range(embedding_mat.shape[0]), batch_size), total=est_batches):
            tail_sim_vals = embedding_mat @ embedding_mat[tail_inds].transpose(1, 0)
            meet_threshold = torch.where(tail_sim_vals > threshold)
            pair_iter = zip(meet_threshold[0].tolist(), (meet_threshold[1] + tail_inds[0]).tolist())
            pair_iter = filter(lambda p: p[0] != p[1], pair_iter) # Ignore self-loops
            w.write((
                "\n".join([
                    "\t".join([str(h), rel_id, str(p)]) for h,p in pair_iter
                ]) + '\n'
            ).encode())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])