#!/usr/bin/env python3

# STL
import time
import csv
import argparse
import sys
import functools
from typing import Iterable, List, Any, Generator
# 3rd Party
import psycopg2
from tokenizers.pre_tokenizers import BertPreTokenizer
# local
from gatbert.data import parse_ez_stance, Sample
from gatbert.constants import *
from gatbert.utils import CumProf, DurationLogger


def generate_degree_column(conn):
    conn.autocommit = False
    with conn:
        with conn.cursor() as curs:
            try:
                curs.execute("ALTER TABLE nodes DROP COLUMN IF EXISTS degree")
                curs.execute("ALTER TABLE nodes DROP COLUMN IF EXISTS out_degree")
                curs.execute("ALTER TABLE nodes DROP COLUMN IF EXISTS in_degree")
                curs.execute("ALTER TABLE nodes ADD COLUMN out_degree INTEGER DEFAULT 0")
                curs.execute("ALTER TABLE nodes ADD COLUMN  in_degree INTEGER DEFAULT 0")

                out_degree_comm = "update nodes n set out_degree = subquery.out_degree from (select start_id as id, count(*) as out_degree from edges group by start_id) as subquery where n.id = subquery.id;"
                in_degree_comm = "update nodes n set in_degree = subquery.in_degree from (select end_id as id, count(*) as in_degree from edges group by end_id) as subquery where n.id = subquery.id;"

                curs.execute(out_degree_comm)
                curs.execute(in_degree_comm)
                curs.execute("ALTER TABLE nodes ADD COLUMN degree INTEGER GENERATED ALWAYS AS (in_degree + out_degree) STORED")
            except (Exception, psycopg2.DatabaseError) as err:
                conn.rollback()
                print(err)



def extract(conn, sample_gen: Iterable[Sample], max_hops: int = 2) -> Generator[List[Any], None, None]:

    # Wrap all DB calls in memoized functions
    cursor = conn.cursor()

    @CumProf
    @functools.lru_cache
    def get_uri(node_id):
        cursor.execute(f"SELECT uri FROM nodes WHERE id={node_id}")
        return cursor.fetchall()[0][0]

    @CumProf
    @functools.lru_cache
    def find_match(token):
        token = token.lower()
        cursor.execute(f"SELECT id FROM nodes WHERE uri='/c/en/{token}' ORDER BY degree DESC LIMIT 1")
        # TODO: I'd like to do a regex like this, but it's nearly 2 orders of magnitude slower
        # cursor.execute(f"SELECT id FROM nodes WHERE uri LIKE '/c/en/{token}%' ORDER BY degree DESC LIMIT 1")
        records = cursor.fetchall()
        if records:
            return records[0][0]
        return None

    @DurationLogger
    @functools.lru_cache
    def find_neighbors(node_id):
        cursor.execute(f"SELECT start_id,end_id,relation_id FROM edges WHERE start_id={node_id}")
        records = cursor.fetchall()

        cursor.execute(f"SELECT start_id,end_id,relation_id FROM edges WHERE end_id={node_id}")
        records.extend(cursor.fetchall())

        edges = set()
        for (start_id, end_id, orig_id) in records:
            rel_obj = CN_RELATIONS[orig_id]
            edges.add( (start_id, end_id, rel_obj.internal_id) )
            # Add the reverse edge if such an edge exist
            rev_rel_obj = REV_RELATIONS.get(orig_id)
            if rev_rel_obj:
                edges.add( (end_id, start_id, rev_rel_obj.internal_id) )

        return list(edges)

    pretok = BertPreTokenizer()
    for sample in sample_gen:
        context = sample.context
        target = sample.target
        stance = sample.stance.value

        target_toks = [pair[0] for pair in pretok.pre_tokenize_str(target)]
        context_toks = [pair[0] for pair in pretok.pre_tokenize_str(context)]
        toks = target_toks + context_toks


        edges = []
        kb_nodes = []
        kb_indices = dict()
        def get_node_index(node_id):
            if node_id not in kb_indices:
                kb_indices[node_id] = len(toks) + len(kb_nodes)
                kb_nodes.append(get_uri(node_id))
            return kb_indices[node_id]

        frontier = []
        for (i, tok) in enumerate(toks):
            node_id = find_match(tok)
            if node_id is None:
                continue
            kb_index = get_node_index(node_id)
            # Bidirectional edge between the token and its KB node match
            edges.append((i, kb_index, TOKEN_TO_KB_RELATION_ID))
            edges.append((kb_index, i, TOKEN_TO_KB_RELATION_ID))
            frontier.append((node_id, 0))
        print(f"find_match={find_match.reset()}")

        visited = set()
        while frontier:
            (node_id, hops) = frontier.pop(0)
            visited.add(node_id)
            if hops + 1 <= max_hops:
                for (head_node_id, tail_node_id, rel_id) in find_neighbors(node_id):
                    head_index = get_node_index(head_node_id)
                    tail_index = get_node_index(tail_node_id)
                    edges.append((head_index, tail_index, rel_id))

                    if head_node_id not in visited:
                        frontier.append((head_node_id, hops + 1))
                    if tail_node_id not in visited:
                        frontier.append((tail_node_id, hops + 1))
        print(f"find_neighbors={find_neighbors.reset()}")
        print(f"get_uri={get_uri.reset()}")

        # TODO: Add any additional edges missing among the nodes of the subgraph

        yield [stance, len(target_toks), len(context_toks), len(kb_nodes), *target_toks, *context_toks, *kb_nodes, *edges]


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='Extract subgraphs from ConceptNet for stance samples')

    parser.add_argument("--gen_degree", action="store_true", help="Force regeneration of the degree column for the nodes table")

    parser.add_argument("--ezstance", type=str, metavar="input.csv", help="File containing stance data from the EZStance dataset")
    parser.add_argument("--vast",     type=str, metavar="input.csv", help="File containing stance data from the VAST dataset")
    parser.add_argument("--semeval",  type=str, metavar="input.txt", help="File containing stance data from SemEval2016-Task6")
    parser.add_argument("-o",         type=str, required=True, metavar="output_file.tsv", help="File pretokenized stance samples, with graph nodes")
    args = parser.parse_args(raw_args)

    conn = psycopg2.connect(dbname='conceptnet5', host='127.0.0.1')

    if args.gen_degree:
        generate_degree_column(conn)
        conn.close()
        return

    if not any([args.ezstance, args.vast, args.semeval]):
        print("Must select one of --ezstance, --vast, or --semeval", file=sys.stderr)
        sys.exit(1)
    if args.ezstance:
        sample_gen = parse_ez_stance(args.ezstance)
    elif args.vast:
        raise RuntimeError("--vast not yet supported")
    else:
        raise RuntimeError("--semeval not yet supported")

    tagged_samples = extract(conn, sample_gen)
    tagged_samples = map(lambda s: [str(el) for el in s], tagged_samples)
    with open(args.o, 'w', encoding='utf-8') as w:
        writer = csv.writer(w, delimiter='\t')
        writer.writerows(tagged_samples)
    conn.close()

if __name__ == "__main__":
    main(sys.argv[1:])
