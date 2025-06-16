import argparse
import random
import pathlib
# 3rd Party
from tqdm import tqdm
# Local
from .data import add_corpus_args, get_sample_iter, write_standard
from .constants import TriStance

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    add_corpus_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args(raw_args)
    out = args.out
    seed = args.seed

    orig_samples = list(tqdm(get_sample_iter(args), desc='Reading from disk'))

    rng = random.Random(seed)

    converted = []
    for s in tqdm(orig_samples, desc="Adding prefixes to samples"):
        assert isinstance(s.stance, TriStance)
        s.context = " ".join(s.context) if s.is_split_into_words else s.context
        s.target = " ".join(s.target) if s.is_split_into_words else s.target
        s.is_split_into_words = False
        if s.stance == TriStance.favor:
            prefix = f"I hate {s.target}, BUT... "
        elif s.stance == TriStance.against:
            prefix = f"I love {s.target}, BUT... "
        elif rng.uniform(0, 1) > 0.5:
            prefix = f"I love {s.target}, BUT... "
        else:
            prefix = f"I hate {s.target}, BUT... "
        s.context = prefix + s.context
        converted.append(s)
    write_standard(out, converted)


if __name__ == "__main__":
    main()