"""
Makes 'pruned_nodes' and 'prune_edges' tables in conceptnet5
"""
# STL
import sys
import argparse
# 3rd Party
import psycopg2
# Local
from .constants import DEFAULT_PG_ARGS

def generate_degree_column(conn):
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
            raise err
    conn.commit()

def make_pruned_nodes(conn):
    with conn.cursor() as curs:
        try:
            curs.execute("DROP TABLE IF EXISTS pruned_nodes")
            curs.execute("CREATE TABLE pruned_nodes (LIKE nodes INCLUDING ALL)")
            curs.execute("ALTER TABLE pruned_nodes DROP COLUMN IF EXISTS degree")
            curs.execute("ALTER TABLE pruned_nodes DROP COLUMN IF EXISTS out_degree")
            curs.execute("ALTER TABLE pruned_nodes DROP COLUMN IF EXISTS in_degree")
            curs.execute("INSERT INTO pruned_nodes(id, uri) SELECT id,uri FROM nodes WHERE uri LIKE '/c/en/%'")
        except (Exception, psycopg2.DatabaseError) as err:
            conn.rollback()
            raise err
    conn.commit()

def make_pruned_edges(conn):
    with conn.cursor() as curs:
        try:
            curs.execute("SELECT COUNT(ed.id) FROM edges ed INNER JOIN pruned_nodes start ON ed.start_id = start.id INNER JOIN pruned_nodes stop ON ed.end_id = stop.id")
            curs.execute("DROP TABLE IF EXISTS pruned_edges")
            curs.execute("CREATE TABLE pruned_edges (LIKE edges INCLUDING ALL)")
            curs.execute("ALTER TABLE pruned_edges DROP weight")
            curs.execute("ALTER TABLE pruned_edges DROP data")
            curs.execute("ALTER TABLE pruned_edges DROP uri")
            curs.execute("INSERT INTO pruned_edges(id, relation_id, start_id, end_id) SELECT ed.id, ed.relation_id, ed.start_id, ed.end_id FROM edges ed INNER JOIN pruned_nodes start ON ed.start_id = start.id INNER JOIN pruned_nodes stop ON ed.end_id = stop.id")
        except (Exception, psycopg2.DatabaseError) as err:
            conn.rollback()
            raise err
    conn.commit()



def main(raw_args=None):

    parser = argparse.ArgumentParser(description="Add supplemental tables and columns to CN's Postgres database to speed up subgraph extraction")

    parser.add_argument("-pg", default=DEFAULT_PG_ARGS, metavar=DEFAULT_PG_ARGS, help="Arguments for the psycopg2 connection object")
    parser.add_argument("--nodes", action="store_true", help="Create 'pruned_nodes' table")
    parser.add_argument("--edges", action="store_true", help="Create 'pruned_edges' table. Requires --nodes to have already been run")
    parser.add_argument("--all", action="store_true", help="Run (or re-run) --nodes and --edges")

    args = parser.parse_args(raw_args)

    conn = psycopg2.connect(args.pg)
    conn.autocommit = False

    if args.all:
        make_pruned_nodes(conn)
        make_pruned_edges(conn)
    elif args.nodes or args.edges:
        if args.nodes:
            make_pruned_nodes(conn)
        if args.edges:
            with conn.cursor() as curs:
                curs.execute("SELECT EXISTS (SELECT 1 FROM pg_tables WHERE tablename='pruned_nodes')")
                result = curs.fetchall()[0][0]
            if not result:
                print("Pruned_nodes doesn't exist, running --nodes routine first...")
                make_pruned_nodes(conn)
            make_pruned_edges(conn)
    else:
        print("Specify one of --nodes, --edges, or --all to do something")
        sys.exit(1)



if __name__ == "__main__":
    main(sys.argv[1:])