#!/usr/bin/env python3

# STL
import time
import csv
import argparse
import sys
import functools
from typing import Iterable, List, Any, Generator
from collections import defaultdict
# 3rd Party
import psycopg2
from tokenizers.pre_tokenizers import BertPreTokenizer
# local
from gatbert.data import parse_ez_stance, Sample
from gatbert.constants import *
from gatbert.utils import CumProf, DurationLogger, batched

def batched_fetch(conn, n):
    while True:
        duration = -time.time()
        batch = conn.fetchmany(n)
        if not batch:
            break
        duration += time.time()
        print(f"Fetched {len(batch)} records in {duration} seconds")
        yield from batch

def timed_query(cursor, query):
    duration = -time.time()
    cursor.execute(query)
    recs = cursor.fetchall()
    duration += time.time()
    print(f"Fetched {len(recs)} records in {duration} seconds")
    return recs


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

    def find_match(token):
        token = token.lower()
        cursor.execute(f"SELECT id,uri FROM nodes WHERE uri='/c/en/{token}' ORDER BY degree DESC LIMIT 1")
        # TODO: I'd like to do a regex like this, but it's nearly 2 orders of magnitude slower
        # cursor.execute(f"SELECT id FROM nodes WHERE uri LIKE '/c/en/{token}%' ORDER BY degree DESC LIMIT 1")
        records = cursor.fetchall()
        if records:
            return records[0]
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
    def preprocess(sample: Sample):
        return [sample.stance.value,
                [pair[0] for pair in pretok.pre_tokenize_str(sample.target)],
                [pair[0] for pair in pretok.pre_tokenize_str(sample.context)]
        ]
    samples = list(map(preprocess, sample_gen))[:50]

    
    tok2id = {}
    # Don't try to find matches for these--wil break the query.
    tok2id["'"] = None
    tok2id['"'] = None
    duration = -time.time()
    toks = list(tok for s in samples for tok in s[1] + s[2])
    for tok in toks:
        if tok not in tok2id:
            tok2id[tok] = find_match(tok)
    duration += time.time()
    print(f"Queried {len(toks)} tokens in {duration} seconds")
    tok2id = {k:v for (k,v) in tok2id.items() if v is not None}

    # head_id -> {(tail_id, rel_id)}
    adj = defaultdict(set)

    visited = set()
    frontier = {rec[0] for rec in tok2id.values()}
    cursor.execute("CREATE TEMP TABLE query_ids (id INT PRIMARY KEY)")
    for i in range(1, max_hops + 1):
        for id_batch in batched(frontier, 2000):
            values_str = ','.join([f"({id})" for id in id_batch])
            duration = -time.time()
            cursor.execute(f"INSERT INTO query_ids VALUES {values_str}")
            duration += time.time()
            print(f"Inserted {len(id_batch)} in {duration} seconds")

        visited |= frontier
        frontier = set()

        records = set(timed_query(cursor, f"SELECT start_id,end_id,relation_id FROM edges ed JOIN query_ids q ON (ed.end_id = q.id OR ed.start_id = q.id)"))
        # records = set(timed_query(cursor, f"SELECT start_id,end_id,relation_id FROM edges ed JOIN query_ids q ON ed.end_id = q.id"))
        # records |= set(timed_query(cursor, f"SELECT start_id,end_id,relation_id FROM edges ed JOIN query_ids q ON ed.start_id = q.id"))

        duration = -time.time()
        i = 0
        for (start_id, end_id, orig_id) in records:
            forward_rel = CN_RELATIONS[orig_id]
            adj[start_id].add((end_id, forward_rel.internal_id))
            # Add reverse edge
            if forward_rel.directed:
                reverse_rel = REV_RELATIONS[orig_id]
                adj[end_id].add((start_id, reverse_rel.internal_id))
            else:
                adj[end_id].add((start_id, forward_rel.internal_id))
            if start_id not in visited:
                frontier.add(start_id)
            if end_id not in visited:
                frontier.add(end_id)
            i += 1
        duration += time.time()
        print(f"{i} edges in {duration} seconds")

        cursor.execute("TRUNCATE TABLE query_ids")

    # Loop through the tokens
    # Get the token's assigned CN node
    # Take N hops away from that CN node to get additional neighbors
    # TODO: Eventually, patch the graph with missing edges (even if we have all the N-hop neighbors).
    # TODO: We could do this by keeping a list of "outstanding" edges that could be added after the fact
    # Convert KB ids in sample to KB uris

    return

def main(raw_args=None):
    parser = argparse.ArgumentParser(description='Extract subgraphs from ConceptNet for stance samples')

    parser.add_argument("--gen_degree", action="store_true", help="Force regeneration of the degree column for the nodes table")

    parser.add_argument("--ezstance", type=str, metavar="input.csv", help="File containing stance data from the EZStance dataset")
    parser.add_argument("--vast",     type=str, metavar="input.csv", help="File containing stance data from the VAST dataset")
    parser.add_argument("--semeval",  type=str, metavar="input.txt", help="File containing stance data from SemEval2016-Task6")
    parser.add_argument("-o",         type=str, required=True, metavar="output_file.tsv", help="File pretokenized stance samples, with graph nodes")
    args = parser.parse_args(raw_args)

    conn = psycopg2.connect(dbname='conceptnet5', host='127.0.0.1')
    # conn = psycopg2.connect(dbname='conceptnet5', host='/home/ethanlmines/blue_dir/repos/EthansGuides/conceptnet/pg_sockets/')

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
