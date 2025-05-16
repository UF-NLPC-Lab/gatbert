import csv
# 3rd Party
from tqdm import tqdm
# Local
from .data import CORPUS_PARSERS, parse_vast, get_en_pipeline, pretokenize_cn_uri
from .utils import Dictionary

def main():

    parser = parse_vast
    csv_path = "/home/ethanlmines/blue_dir/datasets/VAST/vast_train.csv"
    cn_path  = "/home/ethanlmines/blue_dir/datasets/conceptnet/luo/assertions.tsv"
    out_path = "./temp/filter_graph.tsv"

    # csv_path = "/home/ethanlmines/blue_dir/datasets/VAST/vast_train.csv"
    samples = list(parser(csv_path))
    pipeline = get_en_pipeline()

    tags = {"NOUN", "PNOUN", "ADJ", "ADV"}

    d = Dictionary()
    for sample in tqdm(samples):
        d.update([t.lemma_ for t in pipeline(sample.context) if t.pos_ in tags])
    top_lemmas = d.filter_extremes(no_below=2, no_above=0.5, keep_tokens=5000)

    filtered_rows = []
    with open(cn_path, 'r') as r:
        rows = list(csv.reader(r, delimiter='\t'))
    for row in tqdm(rows):
        head = pretokenize_cn_uri(row[2])
        tail = pretokenize_cn_uri(row[3])
        if (len(head) == 1 and head[0] in top_lemmas) or (len(tail) == 1 and tail[0] in top_lemmas):
            filtered_rows.append(row)
            if len(filtered_rows) >= 100:
                break
    with open(out_path, 'w') as w:
        writer = csv.writer(w, delimiter='\t')
        writer.writerows(filtered_rows)


if __name__ == "__main__":
    main()