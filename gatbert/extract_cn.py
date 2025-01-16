#!/usr/bin/env python3

# STL
import json
import time
import argparse
import sys
from typing import Iterable, Dict, Any 
from collections import defaultdict
# 3rd Party
import psycopg2
# local
from .data import parse_ez_stance, Sample, PretokenizedSample, get_default_pretokenize
from .constants import *
from .utils import batched

def extract(conn, sample_gen: Iterable[PretokenizedSample], max_hops: int = 2) -> Dict[str, Any]:

    # Wrap all DB calls in memoized functions
    cursor = conn.cursor()
    # Temporary table used to speed up queries via inner joins
    cursor.execute("CREATE TEMP TABLE query_ids (id INT PRIMARY KEY)")

    def fill_query_table(ids):
        for id_batch in batched(ids, 2000):
            values_str = ','.join([f"({id})" for id in id_batch])
            cursor.execute(f"INSERT INTO query_ids VALUES {values_str}")

    ############ Find 'zero-hop neighbor' (single matching node for a token) ##########################3
    tok2id = {}
    # Don't try to find matches for these don't need them and requires extra sanitization in the query
    tok2id["'"] = None
    tok2id['"'] = None
    duration = -time.time()
    toks = list(tok.lower() for s in sample_gen for tok in s.context + s.target)
    for tok in toks:
        if tok not in tok2id:
            # TODO: I'd like to do a regex instead, but it's slower
            cursor.execute(f"SELECT id FROM pruned_nodes WHERE uri='/c/en/{tok.lower()}'")
            records = cursor.fetchall()
            tok2id[tok] = records[0][0] if records else None
    duration += time.time()
    print(f"Queried {len(toks)} tokens in {duration} seconds")
    tok2id = {k:v for (k,v) in tok2id.items() if v is not None}

    ##################### Find max_hops-neighbors from those matched nodes, along with their edges ###################3
    adj = defaultdict(set)
    def add_edge(start_id, end_id, orig_id):
        forward_rel = CN_RELATIONS[orig_id]
        adj[start_id].add((end_id, forward_rel.internal_id))
        # Add reverse edge
        if forward_rel.directed:
            reverse_rel = REV_RELATIONS[orig_id]
            adj[end_id].add((start_id, reverse_rel.internal_id))
        else:
            adj[end_id].add((start_id, forward_rel.internal_id))

    id2uri = {}
    # head_id -> {(tail_id, rel_id)}
    visited = set()
    frontier = set(tok2id.values())
    for _ in range(1, max_hops + 1):
        fill_query_table(frontier)

        ################################################################## Query the Edge Table #######################################################
        print(f"Querying with {len(frontier)} node IDs")
        cursor.execute("SELECT start_id,end_id,relation_id FROM pruned_edges ed JOIN query_ids q ON (ed.end_id = q.id OR ed.start_id = q.id)")
        # We use a set here because many nodes have self-loops, meaning this query will return a few duplicate edges
        # The self-loops are almost all synonym relations. In general I'd call them light noise.
        records = set(cursor.fetchall())
        print(f"Fetched {len(records)} edges")

        ################################ Add Edges to Adj matrix and next-hop neighbors to frontier #####################################################
        visited |= frontier
        frontier = set()
        for (start_id, end_id, orig_id) in records:
            add_edge(start_id, end_id, orig_id)
            if start_id not in visited:
                frontier.add(start_id)
            if end_id not in visited:
                frontier.add(end_id)

        ################################# Get the URIs for the relevant node ids ##############################################
        cursor.execute("SELECT n.id,uri FROM pruned_nodes n JOIN query_ids q ON n.id = q.id;")
        id2uri.update(cursor.fetchall())
        cursor.execute("TRUNCATE TABLE query_ids")

    # The last nodes we encountered may have edges between them as well. Probability a rarity, but just to be safe...
    fill_query_table(frontier)
    print(f"Querying with {len(frontier)} node IDs")
    cursor.execute("SELECT start_id,end_id,relation_id FROM pruned_edges ed JOIN query_ids q_start ON ed.start_id = q_start.id JOIN query_ids q_end ON ed.end_id = q_end.id;")
    records = set(cursor.fetchall())
    print(f"Adding {len(records)} additional edges but no additional nodes")
    [add_edge(*rec) for rec in records]

    # Also need the URIs for those final nodes too
    cursor.execute("SELECT n.id,uri FROM pruned_nodes n JOIN query_ids q ON n.id = q.id;")
    id2uri.update(cursor.fetchall())

    graph_dict = {
        "tok2id" : tok2id,
        "id2uri" : id2uri,
        "adj": {k:list(v) for k,v in adj.items()}
    }
    return graph_dict

def main(raw_args=None):
    parser = argparse.ArgumentParser(description='Extract subgraphs from ConceptNet for stance samples')

    parser.add_argument("-pg", default=DEFAULT_PG_ARGS, metavar=DEFAULT_PG_ARGS, help="Arguments for the psycopg2 connection object")
    parser.add_argument("--ezstance", type=str, metavar="input.csv", help="File containing stance data from the EZStance dataset")
    parser.add_argument("--vast",     type=str, metavar="input.csv", help="File containing stance data from the VAST dataset")
    parser.add_argument("--semeval",  type=str, metavar="input.txt", help="File containing stance data from SemEval2016-Task6")
    parser.add_argument("-o",         type=str, required=True, metavar="output_file.json", help="File holding CN subgraph data")
    args = parser.parse_args(raw_args)

    conn = psycopg2.connect(args.pg)

    if not any([args.ezstance, args.vast, args.semeval]):
        print("Must select one of --ezstance, --vast, or --semeval", file=sys.stderr)
        sys.exit(1)
    if args.ezstance:
        sample_gen = parse_ez_stance(args.ezstance)
    elif args.vast:
        raise RuntimeError("--vast not yet supported")
    else:
        raise RuntimeError("--semeval not yet supported")
    
    pretok_func = get_default_pretokenize()
    sample_gen = map(pretok_func, sample_gen)
    graph = extract(conn, sample_gen)
    conn.close()
    with open(args.o, 'w', encoding='utf-8') as w:
        json.dump(graph, w)

if __name__ == "__main__":
    main(sys.argv[1:])
