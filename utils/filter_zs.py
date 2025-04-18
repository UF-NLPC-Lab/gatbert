#!/usr/bin/env python3
import sys
import csv

reader = csv.DictReader(sys.stdin)
writer = csv.DictWriter(sys.stdout, fieldnames=reader.fieldnames)
writer.writeheader()

for row in filter(lambda r: int(r['seen?']) == 0, reader):
    writer.writerow(row)
