import os
import io
import gzip
import logging
import operator
from functools import reduce
from collections import Counter
from typing import List, Any, Iterable
import time
from contextlib import contextmanager

class Dictionary:
    """
    Emulation of Gensim Dictionary's main functionality
    """
    def __init__(self):
        self.__dfs = Counter()
        self.__n_docs = 0
    def update(self, doc: Iterable[str]):
        terms = list(doc)
        self.__dfs.update(set(terms))
        self.__n_docs += 1
    def filter_extremes(self, no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None):
        abs_no_above = int(no_above * self.__n_docs) if no_above is not None else self.__n_docs
        no_below = 0 if no_below is None else no_below
        good_ids = [term for term,count in self.__dfs.items() if no_below <= count <= abs_no_above]
        good_ids = sorted(good_ids, key=lambda x: self.__dfs[x], reverse=True)
        if keep_n is not None:
            good_ids = good_ids[:keep_n]
        return good_ids

def exists_gzip_or_plain(path: os.PathLike):
    str_path = str(path)
    gz_path = str_path if str_path.endswith(".gz") else str_path + ".gz"
    short_path = gz_path[:-3]
    return os.path.exists(gz_path) or os.path.exists(short_path)

class GzipWrapper:
    def __init__(self, f):
        self.f = f
    def write(self, str_data):
        return self.f.write(str_data.encode())

@contextmanager
def open_gzip_or_plain(path: os.PathLike):
    str_path = str(path)
    gz_path = str_path if str_path.endswith(".gz") else str_path + ".gz"
    short_path = gz_path[:-3]

    if os.path.exists(gz_path):
        with gzip.open(gz_path, 'rb') as f:
            try:
                buffer = io.StringIO()
                buffer.write(f.read().decode())
                buffer.seek(0)
                yield buffer
            finally:
                buffer.close()
    else:
        with open(short_path, 'r') as f:
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