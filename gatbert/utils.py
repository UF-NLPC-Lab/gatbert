from functools import reduce
from typing import List, Any
import time

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

class CumProf:
    def __init__(self, func):
        self.func = func
        self.cumtime = 0
        self.count = 0
    def reset(self):
        printout = str(self)
        self.cumtime = 0
        self.count = 0
        return printout
    def __call__(self, *args, **kwargs):
        start = time.time()
        rval = self.func(*args, **kwargs)
        self.cumtime += (time.time() - start)
        self.count += 1
        return rval
    def __str__(self):
        return f"CumProf(cumtime={self.cumtime}, count={self.count})"

def DurationLogger(f):
    def wrapped(*args, **kwargs):
        duration = -time.time()
        rval = f(*args, **kwargs)
        duration += time.time()
        print(f"Took {duration} seconds")
        return rval
    return wrapped