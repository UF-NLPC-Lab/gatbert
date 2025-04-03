# STL
import sys
import pathlib
# Local
from .graph import *
from .constants import SpecialRelation
from .utils import exists_gzip_or_plain

if __name__ == "__main__":
    graph_dir = pathlib.Path(sys.argv[1])
    id2ent = {v:k for k,v in read_entitites(get_entities_path(graph_dir)).items()}
    id2rel = {v:k for k,v in read_relations(get_relations_path(graph_dir)).items()}
    id2rel[SpecialRelation.KB_SIM.value] = "KB_SIM"

    adj = read_adj_mat(get_triples_path(graph_dir), make_inverse_rels=False)
    bert_triples_path = get_bert_triples_path(graph_dir)
    if exists_gzip_or_plain(bert_triples_path):
        bert_adj = read_bert_adj_mat(bert_triples_path, sim_threshold=0)
        update_adj_mat(adj, bert_adj)

    for (head, edges) in adj.items():
        head_name = id2ent[head]
        for (tail, rel) in edges:
            tail_name = id2ent[tail]
            rel_name = id2rel[rel]
            print("DUMMY", rel_name, head_name, tail_name, sep='\t')