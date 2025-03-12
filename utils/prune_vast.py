#!/usr/bin/env python3
"""
Prunes the seen-topics from the dev or test set of VAST
"""
import sys
import csv

reader = csv.reader(sys.stdin, quotechar='"')

header = next(reader)
seen_field = header.index("seen?")

writer = csv.writer(sys.stdout, quotechar='"')
writer.writerow(header)

valid_rows = filter(lambda r: int(r[seen_field]) == 0, reader)
writer.writerows(valid_rows)