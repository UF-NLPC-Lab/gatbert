import pathlib
import argparse
from itertools import islice
# 3rd Party
from nltk.wsd import lesk
from tqdm import tqdm
# Local
from .cn import CN
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

seen = dict()
def cn_expand_sample(cn: CN, sample: Sample):
    lang = sample.lang or 'en'
    assert lang == 'en', "Only English currently supported"
    pipe = SPACY_PIPES[lang]
    target = " ".join(sample.target) if sample.is_split_into_words else sample.target
    target_doc = pipe(target)

    simple_str = " ".join([t.lemma_ for t in target_doc])
    if lang == 'en':
        simple_str = simple_str.replace(" 's", "'s")
    simple_str = simple_str.replace(" ", "_")


    alt_targets = []
    if simple_str in seen:
        alt_targets = seen[simple_str]
    elif simple_str in cn.node2id:
        node_id = cn.node2id[simple_str]
        syn_rel_id = cn.relation2id['/r/Synonym']

        syn_names = set(
            cn.id2node[neigh_id]
            for (rel_id, neigh_id) in cn.adj.get(node_id, []) + cn.rev_adj.get(node_id, [])
            if rel_id == syn_rel_id
        )
        alt_targets = [n for n in syn_names if n != simple_str]
        seen[simple_str] = alt_targets

    context_str = " ".join(sample.context) if sample.is_split_into_words else sample.context
    for t in alt_targets:
        yield Sample(context=context_str,
                     target=t.replace("_", " "),
                     stance=sample.stance,
                     is_split_into_words=False,
                     lang=lang)

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    add_corpus_args(parser)
    parser.add_argument("-o", metavar="out.tsv", required=True, help="Output path to write samples (in --standard format)")
    parser.add_argument("-N", metavar="8", type=int, default=8, help="Maximum silver samples to make for a gold sample")
    parser.add_argument("--alg", metavar='wn|cn', default='wn')
    parser.add_argument('-cn', type=pathlib.Path)
    args = parser.parse_args(raw_args)
    sample_iter = get_sample_iter(args)
    out_path = args.o
    max_new = args.N

    if args.alg == 'wn':
        expand_func = expand_sample
    elif args.alg == 'cn':
        assert args.cn
        cn = CN.load(args.cn)
        expand_func = lambda s: cn_expand_sample(cn, s)

    def combined_iter():
       for sample in tqdm(sample_iter):
           yield from islice(expand_func(sample), max_new)
    write_standard(out_path, combined_iter())

if __name__ == "__main__":
    main()