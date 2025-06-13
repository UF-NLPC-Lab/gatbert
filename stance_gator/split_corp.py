import argparse
import copy
import pathlib

from tqdm import tqdm

from .data import SPACY_PIPES, add_corpus_args, get_sample_iter, write_standard

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    add_corpus_args(parser)
    parser.add_argument("-o", type=pathlib.Path, required=True)
    args = parser.parse_args(raw_args)

    orig_samples = list(tqdm(get_sample_iter(args), desc='Reading from disk'))

    new_samples = []
    for s in tqdm(orig_samples, desc='Splitting Samples'):
        s.context = " ".join(s.context) if s.is_split_into_words else s.context
        s.target = " ".join(s.target) if s.is_split_into_words else s.target
        s.is_split_into_words = False

        lang = s.lang or 'en'
        pipe = SPACY_PIPES[lang]
        doc = pipe(s.context)
        sents = list(doc.sents)
        if len(sents) > 1:
            for sent in doc.sents:
                s_copy = copy.copy(s)
                s_copy.context = str(sent)
                new_samples.append(s_copy)
    new_samples.extend(orig_samples)
    write_standard(args.o, tqdm(new_samples, desc=f'Writing samples to {args.o}'))

if __name__ == "__main__":
    main()