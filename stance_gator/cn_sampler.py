# STL
import argparse
import csv
import functools
import random
# 3rd Party
from tqdm import tqdm
# Local
from .data import pretokenize_cn_uri
from .constants import TriStance


POS_RELATIONS = {
    "/r/Synonym",
    "/r/IsA",
    "/r/HasProperty",
    "/r/Desires"
}
NEG_RELATIONS = {"/r/Antonym",
                 "/r/NotHasProperty",
                 "/r/NotDesires",
                 "/r/DistinctFrom",
                 "/r/NotCapableOf",
                 }

def main():

    def parse_ratio(ratio):
        ratio = tuple(float(x) for x in ratio.split(','))
        assert abs(sum(ratio) - 1) < 1e-6
        return ratio

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="assertions.csv")
    parser.add_argument("-N", type=int, default=10000, metavar="10000")
    parser.add_argument("-o", metavar="data.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    max_samples = args.N
    cn_path = args.i
    seed = args.seed
    all_entities = set()

    pos_samples = []
    neg_samples = []
    related_samples = []

    @functools.cache
    def ent_pred(ent):
        simp = "".join(pretokenize_cn_uri(ent))
        if simp.rstrip('s').isnumeric():
            return False
        return True
    def same_ent(head, tail):
        head_tok = "".join(pretokenize_cn_uri(head))
        tail_tok = "".join(pretokenize_cn_uri(tail))
        if head_tok == tail_tok:
            return True
        return False

    with open(cn_path, 'r') as r:
        rows = list(csv.reader(r, delimiter='\t'))

    nrows = len(rows)
    rows = tqdm(rows, total=nrows)
    rows = filter(lambda r: ent_pred(r[2]) and ent_pred(r[3]) and not same_ent(*r[2:4]), rows)
    for row in rows: 
        rel, head, tail = row[1:4]
        all_entities.add(head)
        all_entities.add(tail)
        sample = (head, tail)
        if rel in POS_RELATIONS:
            pos_samples.append(sample)
        elif rel in NEG_RELATIONS:
            neg_samples.append(sample)
        else:
            related_samples.append(sample)

    rng = random.Random(seed)
    rng.shuffle(pos_samples)
    rng.shuffle(neg_samples)
    rng.shuffle(related_samples)

    written = 0
    with open(args.o, 'w') as w:
        writer = csv.writer(w)
        writer.writerow(["Stance", "Head", "Tail"])

        # Round-robin allocate samples from each class
        while written < max_samples:
            if pos_samples:
                writer.writerow([TriStance.favor.value, *pos_samples.pop()])
                written += 1
            else:
                break
            if neg_samples and written < max_samples:
                writer.writerow([TriStance.against.value, *neg_samples.pop()])
                written += 1
            else:
                break
            if related_samples and written < max_samples:
                writer.writerow([TriStance.neutral.value, *related_samples.pop()])
                written += 1
            else:
                break

if __name__ == "__main__":
    main()