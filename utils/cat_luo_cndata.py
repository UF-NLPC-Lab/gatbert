#!/usr/bin/env python3
from collections import OrderedDict
import sys

seen_triples = OrderedDict()

for l in sys.stdin:
    [head, tail, rel] = l.split()
    # Replicate their bug of only keeping the last relation seen for that (head, tail) pair
    seen_triples[head, tail] = (head, rel, tail)
for (head, rel, tail) in seen_triples.values():
    print(f"DUMMY\t/r/{rel}\t/c/en/{head}\t/c/en/{tail}")