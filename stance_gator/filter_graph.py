import csv
import argparse
# 3rd Party
from tqdm import tqdm
# Local
from .data import parse_ez_stance, parse_vast, get_en_pipeline, pretokenize_cn_uri, extract_lemmas
from .utils import Dictionary

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--vast", metavar="vast_train.csv")
    parser.add_argument("--ezstance", metavar="raw_train_all_onecol.csv")
    parser.add_argument("-cn", metavar="assertions.tsv", required=True)
    parser.add_argument("-o", metavar="filtered_assertions.tsv", required=True)

    args = parser.parse_args(raw_args)
    # csv_path = "/home/ethanlmines/blue_dir/datasets/VAST/vast_train.csv"
    cn_path  = args.cn #"/home/ethanlmines/blue_dir/datasets/conceptnet/luo/assertions.tsv"
    out_path = args.o #"./temp/filter_graph.tsv"
    if args.vast:
        samples = parse_vast(args.vast)
    elif args.ezstance:
        samples = parse_ez_stance(args.ezstance)
    else:
        raise ValueError("Provide --vast or --ezstance")
    samples = list(samples)[:100]
    pipeline = get_en_pipeline()

    d = Dictionary()
    for sample in tqdm(samples, desc="Extracting seeds"):
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