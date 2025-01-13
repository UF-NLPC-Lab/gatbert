#!/usr/bin/env python3

# STL
import argparse
import pdb
import sys
import dataclasses
import functools
# 3rd Party
import psycopg2
import psycopg2.extras
from tokenizers.pre_tokenizers import BertPreTokenizer
# local
from gatbert.data import Sample, parse_ez_stance
from gatbert.constants import CN_RELATIONS, REV_RELATIONS



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

@dataclasses.dataclass
class GraphSample:
    pass

def extract(conn, sample_gen, N=2):

    pretok = BertPreTokenizer()

    cursor = conn.cursor('cursor_unique_name', psycopg2.extras.DictCursor)

    @functools.lru_cache
    def find_match(token):
        token = token.lower()
        cursor.execute("SELECT id,uri FROM nodes ORDER BY degree DESC LIMIT 1")
        records = cursor.fetchall()
        if records:
            return (records[0]['id'], records[0]['uri'])
        return None

    @functools.lru_cache
    def find_neighbors(node_id):
        cursor.execute("SELECT start_id,end_id,relation_id FROM edges WHERE start_id=node_id OR end_id=node_id")
        records = cursor.fetchall()

        neighbors = set()
        edges = {}
        
        for rec in records:
            orig_id = rec['relation_id']
            start_id = rec['start_id']
            end_id = rec['end_id']
            if start_id != node_id:
                neighbors.add(start_id)
            elif end_id != node_id:
                neighbors.add(end_id)

            rel_obj = CN_RELATIONS[orig_id]
            edges.append( (start_id, end_id, rel_obj.internal_id) )

            # Add the reverse edge if such an edge exist
            rev_rel_obj = REV_RELATIONS.get(orig_id)
            if rev_rel_obj:
                edges.append( (end_id, start_id, rev_rel_obj.internal_id) )

        return neighbors, edges

    for sample in sample_gen:
        context = sample.context
        target = sample.target
        stance = sample.stance.value

        target_toks = pretok.pre_tokenize_str(target)
        N_target = len(target_toks)
        context_toks = pretok.pre_tokenize_str(context)
        N_context = len(context_toks)
        toks = target_toks + context_toks

        for tok in toks:
            match = find_match(tok)
            if not match:
                continue
            (node_id, node_uri) = match

            pass

        yield [stance, N_target, N_context, *toks]


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='Extract subgraphs from ConceptNet for stance samples')

    parser.add_argument("--gen_degree", action="store_true", help="Force regeneration of the degree column for the nodes table")

    parser.add_argument("--ezstance", type=str, required=True, metavar="input.csv", help="File containing stance data from the EZStance dataset")
    parser.add_argument("--vast",     type=str, required=True, metavar="input.csv", help="File containing stance data from the VAST dataset")
    parser.add_argument("--semeval",  type=str, required=True, metavar="input.txt", help="File containing stance data from SemEval2016-Task6")
    parser.add_argument("-o",         type=str, required=True, metavar="output_file.tsv", help="File pretokenized stance samples, with graph nodes")
    args = parser.parse_arg(raw_args)

    if args.gen_degree:
        conn = psycopg2.connect("dbname='conceptnet5'")
        generate_degree_column(conn)
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

if __name__ == "__main__":
    main(sys.argv[1:])
