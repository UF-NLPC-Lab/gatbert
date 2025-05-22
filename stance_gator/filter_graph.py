import csv
import argparse
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
    out_path = args.o 
    samples = list(samples)

    d = Dictionary()
    for sample in tqdm(samples, desc="Extracting seeds"):
        pipeline = SPACY_PIPES[sample.lang or 'en']
        d.update(extract_lemmas(pipeline, sample.context))
    top_lemmas = d.filter_extremes(no_below=2, no_above=0.5, keep_tokens=5000)

    filtered_rows = []
    with open(cn_path, 'r') as r:
        rows = list(csv.reader(r, delimiter='\t'))
    for row in tqdm(rows, desc="Filtering edges..."):
        head = pretokenize_cn_uri(row[2])
        tail = pretokenize_cn_uri(row[3])
        if (len(head) == 1 and head[0] in top_lemmas) or (len(tail) == 1 and tail[0] in top_lemmas):
            filtered_rows.append(row)
    with open(out_path, 'w') as w:
        writer = csv.writer(w, delimiter='\t')
        writer.writerows(filtered_rows)


if __name__ == "__main__":
    main()