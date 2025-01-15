#!/bin/bash

set -x

RUN_COMMAND="psql -d conceptnet5 -c"

########### Pruned Nodes ##################
# $RUN_COMMAND "DROP TABLE IF EXISTS pruned_nodes"
# $RUN_COMMAND "CREATE TABLE pruned_nodes (LIKE nodes INCLUDING ALL)"
# $RUN_COMMAND "ALTER TABLE pruned_nodes DROP COLUMN IF EXISTS degree"
# $RUN_COMMAND "ALTER TABLE pruned_nodes DROP COLUMN IF EXISTS out_degree"
# $RUN_COMMAND "ALTER TABLE pruned_nodes DROP COLUMN IF EXISTS in_degree"
# $RUN_COMMAND "INSERT INTO pruned_nodes(id, uri) SELECT id,uri FROM nodes WHERE uri LIKE '/c/en/%'"

############# Pruned Edges ##################

# $RUN_COMMAND "SELECT COUNT(ed.id) FROM edges ed INNER JOIN pruned_nodes start ON ed.start_id = start.id INNER JOIN pruned_nodes stop ON ed.end_id = stop.id"
# $RUN_COMMAND "DROP TABLE IF EXISTS pruned_edges"
# $RUN_COMMAND "CREATE TABLE pruned_edges (LIKE edges INCLUDING ALL)"
# $RUN_COMMAND "ALTER TABLE pruned_edges DROP weight"
# $RUN_COMMAND "ALTER TABLE pruned_edges DROP data"
# $RUN_COMMAND "ALTER TABLE pruned_edges DROP uri"
# $RUN_COMMAND "INSERT INTO pruned_edges(id, relation_id, start_id, end_id) SELECT ed.id, ed.relation_id, ed.start_id, ed.end_id FROM edges ed INNER JOIN pruned_nodes start ON ed.start_id = start.id INNER JOIN pruned_nodes stop ON ed.end_id = stop.id"
