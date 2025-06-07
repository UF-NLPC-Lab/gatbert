import argparse
from itertools import islice
# 3rd Party
from nltk.wsd import lesk
# Local
from .data import SPACY_PIPES, add_corpus_args, get_sample_iter, write_standard
from .sample import Sample

def expand_sample(sample: Sample):
    lang = sample.lang or 'en'
    assert lang == 'en', "Only English currently supported"
    pipe = SPACY_PIPES[lang]

    target = " ".join(sample.target) if sample.is_split_into_words else sample.target
    target_doc = pipe(target)
    if len(target_doc) != 1:
        return

    context_toks = sample.context if sample.is_split_into_words else sample.context.split()
    synset = lesk(context_toks, target)
    if not synset:
        return

    target_lemma = target_doc[0].lemma_.lower()
    context_str = " ".join(context_toks)
    lemma_iter = synset.lemma_names()
    lemma_iter = filter(lambda l: l.lower() != target_lemma, lemma_iter)
    for lemma in lemma_iter:
        yield Sample(context=context_str,
                     target=lemma.replace('_', ' '),
                     stance=sample.stance,
                     is_split_into_words=False,
                     lang=lang)

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    add_corpus_args(parser)
    parser.add_argument("-o", metavar="out.tsv", required=True, help="Output path to write samples (in --standard format)")
    parser.add_argument("-N", metavar="8", type=int, default=4, help="Maximum silver samples to make for a gold sample")
    args = parser.parse_args(raw_args)
    sample_iter = get_sample_iter(args)
    out_path = args.o
    max_new = args.N

    def combined_iter():
        for sample in sample_iter:
            yield from islice(expand_sample(sample), max_new)
    write_standard(out_path, combined_iter())

if __name__ == "__main__":
    main()