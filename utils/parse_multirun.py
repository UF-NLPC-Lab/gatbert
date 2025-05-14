#!/usr/bin/env python3

import os
import glob
import sys
import csv
import math

def cal_mean(seq):
    return sum(seq) / len(seq)
def cal_std(seq, mean):
    return math.sqrt(sum((x - mean)**2 for x in seq) / len(seq))

skipped = []
print("experiment", "n_trials", "mean_macro_f1", "std_macro_f1", sep=',')
for results_dir in sys.argv[1:]:
    model_name = results_dir
    values = []

    for seed_dir in glob.glob(os.path.join(results_dir, "seed_*")):
        file_path = os.path.join(seed_dir, 'metrics.csv')

        if not os.path.exists(file_path):
            print("Skipped", file_path, file=sys.stderr)
            continue
        test_f1 = -1
        with open(file_path, 'r') as r:
            reader = csv.DictReader(r)
            for row in reader:
                if row.get("test_macro_f1"):
                    test_f1 = float(row["test_macro_f1"])
            values.append(test_f1)
    if not values:
        print("Skipped", results_dir, file=sys.stderr)
        continue
    mean_f1 = cal_mean(values)
    stddev_f1 = cal_std(values, mean_f1)
    print(model_name, len(values), mean_f1, stddev_f1, sep=',')
