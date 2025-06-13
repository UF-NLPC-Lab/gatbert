import argparse
import random
import copy
import pathlib
from collections import OrderedDict
# 3rd Party
from tqdm import tqdm
# Local
from .data import SPACY_PIPES, add_corpus_args, get_sample_iter, write_standard

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    add_corpus_args(parser)

    parser.add_argument("--split-prob", type=float, default=.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratio", default=4.0, type=float, help="Ratio of training docs (not samples) to valid docs")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--out-val", type=pathlib.Path)
    args = parser.parse_args(raw_args)
    out_train = args.out
    out_val = args.out_val
    seed = args.seed
    ratio = args.ratio
    split_prob = args.split_prob

    orig_samples = list(tqdm(get_sample_iter(args), desc='Reading from disk'))

    # The same document gets used in more than one sample.
    # We want to avoid data leakage between training and validation
    # Only using an ordered dict for determinism
    doc2samples = OrderedDict()

    for s in tqdm(orig_samples, desc="Grouping samples by document"):
        s.context = " ".join(s.context) if s.is_split_into_words else s.context
        s.target = " ".join(s.target) if s.is_split_into_words else s.target
        s.is_split_into_words = False

        if s.context not in doc2samples:
            doc2samples[s.context] = []
        doc2samples[s.context].append(s)
    sample_groups = list(doc2samples.values())
    rng = random.Random(seed)

    rng.shuffle(sample_groups)

    if out_val:
        train_frac = ratio / (ratio + 1)
        train_end = int(len(sample_groups) * train_frac)
        unsplit_train_samples = [s for group in sample_groups[:train_end] for s in group]
        val_samples = [s for group in sample_groups[train_end:] for s in group]
        write_standard(out_val, tqdm(val_samples, desc=f'Writing val samples to {out_val}'))
    else:
        unsplit_train_samples = [s for group in sample_groups for s in group]

    train_samples = []
    for s in tqdm(unsplit_train_samples, desc='Splitting Training Samples'):
        if rng.uniform(0, 1) > split_prob:
            train_samples.append(s)
            continue
        lang = s.lang or 'en'
        pipe = SPACY_PIPES[lang]
        doc = pipe(s.context)
        sents = list(doc.sents)
        if len(sents) > 1:
            for sent in doc.sents:
                s_copy = copy.copy(s)
                s_copy.context = str(sent)
                train_samples.append(s_copy)
        else:
            train_samples.append(s)
    write_standard(out_train, tqdm(train_samples, desc=f'Writing train samples to {out_train}'))

if __name__ == "__main__":
    main()