import csv
import argparse
from collections import defaultdict
# 3rd Party
from tqdm import tqdm
# Local
from .data import add_corpus_args, get_sample_iter
from .data import SPACY_PIPES, pretokenize_cn_uri, extract_lemmas
from .utils import Dictionary

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    add_corpus_args(parser)
    parser.add_argument("-cn", metavar="assertions.tsv", required=True)
    parser.add_argument("-o", metavar="filtered_assertions.tsv", required=True)
    args = parser.parse_args(raw_args)

    samples = get_sample_iter(args)

    cn_path  = args.cn 

    syn_maps = defaultdict(lambda: defaultdict(set))
    with open(cn_path, 'r') as r:
        rows = list(csv.reader(r, delimiter='\t'))
    for row in tqdm(rows, desc="Searching assertions for synonyms"):
        head, tail = row[2:4]
        head_lang = head.split('/')[2]
        tail_lang = tail.split('/')[2]
        if head_lang != tail_lang:
            assert row[1] == '/r/Synonym'
            head_toks = pretokenize_cn_uri(head)
            tail_toks = pretokenize_cn_uri(tail)
            if len(head_toks) == 1 and len(tail_toks) == 1:
                head_str = head_toks[0]
                tail_str = tail_toks[0]
                if head_lang == 'en':
                    syn_maps[tail_lang][tail_str].add(head_str)
                else:
                    assert tail_lang == 'en'
                    syn_maps[head_lang][head_str].add(tail_str)
    # Convert to regular dictionaries, and lists instead of sets
    syn_maps = {lang:{lemma:list(syns) for lemma,syns in syn_map.items()} for lang, syn_map in syn_maps.items()}

    d = Dictionary()
    samples = list(samples)
    for sample in tqdm(samples, desc="Extracting seeds"):
        pipeline = SPACY_PIPES[sample.lang or 'en']
        lemmas = extract_lemmas(pipeline, sample.context)
        if sample.lang != 'en':
            lang_syn_map = syn_maps[sample.lang]
            lemmas = sum([lang_syn_map.get(l, []) for l in lemmas], [])
        d.update(lemmas)
    top_lemmas = d.filter_extremes(no_below=2, no_above=0.5, keep_tokens=5000)

    filtered_rows = []
    for row in tqdm(rows, desc="Filtering edges..."):
        head = pretokenize_cn_uri(row[2])
        tail = pretokenize_cn_uri(row[3])
        if (len(head) == 1 and head[0] in top_lemmas) or (len(tail) == 1 and tail[0] in top_lemmas):
            filtered_rows.append(row)
    with open(args.o, 'w') as w:
        writer = csv.writer(w, delimiter='\t')
        writer.writerows(filtered_rows)


if __name__ == "__main__":
    main()