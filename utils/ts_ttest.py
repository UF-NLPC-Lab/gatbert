#!/usr/bin/env python3

import os
import glob
import sys
import csv
import math

import numpy as np
from scipy.stats import ttest_rel

def get_values(results_dir):
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
    assert values
    return values, np.mean(values), np.std(values)

candidate = sys.argv[1]
cand_vals, cand_mean, cand_std = get_values(candidate)

print("model", "mean", "std", "p", sep=',')
print(candidate, f"{cand_mean:.3f}", f"{cand_std:.3f}", "")

for results_dir in sys.argv[2:]:
    vals, mean, std = get_values(results_dir)
    res = ttest_rel(cand_vals, vals, alternative='greater')
    print(results_dir, f"{mean:.3f}", f"{std:.3f}", f"{res.pvalue:.3f}")
