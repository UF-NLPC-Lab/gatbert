from typing import Iterable
import pathlib
import os
import gzip
import logging
import operator
from functools import reduce
from typing import List, Any
import time
from contextlib import contextmanager

@contextmanager
def open_gzip_or_plain(path: os.PathLike, mode='r'):
    str_path = str(path)
    gz_path = str_path if str_path.endswith(".gz") else str_path + ".gz"
    short_path = str_path[:-3]

    if os.path.exists(gz_path):
        with gzip.open(gz_path, mode=mode) as f:
            f = map(lambda row: row.decode(), f)
            try:
                yield f
            finally:
                pass
    else:
        with open(short_path, mode=mode) as f:
            try:
                yield f
            finally:
                pass

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def map_func_gen(f, func):
    def mapped(*args, **kwargs):
        return map(f, func(*args, **kwargs))
    return mapped

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

@contextmanager
def change_log_level(logger_name: str, log_level: int = logging.ERROR):
    try:
        logger = logging.getLogger(logger_name)
        old_level = logger.getEffectiveLevel()
        logger.setLevel(log_level)
        yield
    finally:
        logger.setLevel(old_level)

@contextmanager
def time_block(block_name: str):
    try:
        duration = -time.time()
        yield
    finally:
        duration += time.time()
        print(f"{block_name} took {duration} seconds")

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