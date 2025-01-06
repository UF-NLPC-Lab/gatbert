from functools import reduce
from typing import List, Any

def flat_map(f, iterable) -> List[Any]:
    return reduce(lambda a,b: a + b, map(f, iterable), [])

def limit_n(iterable, n):
    for (i, el) in enumerate(iterable, start=1):
        yield el
        if i == n:
            break

def batched(iterable, n: int):
    batch = []
    for el in iterable:
        batch.append(el)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch