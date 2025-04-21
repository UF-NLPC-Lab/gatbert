#!/usr/bin/env python3

import os
import sys
import csv

skipped = []
print("experiment", "val_macro_f1", "test_macro_f1", sep=',')
for results_dir in sys.argv[1:]:
    file_path = os.path.join(results_dir, 'metrics.csv')
    if not os.path.exists(file_path):
        skipped.append(results_dir)
        continue
    dirname = os.path.basename(results_dir)
    
    test_f1 = -1
    val_f1 = -1
    
    with open(file_path, 'r') as r:
        reader = csv.DictReader(r)
        for row in reader:
            if row["val_macro_f1"]:
                val_f1 = max(val_f1, float(row["val_macro_f1"]))
            elif row.get("test_macro_f1"):
                test_f1 = float(row["test_macro_f1"])
    print(dirname, val_f1, test_f1, sep=',')
if skipped:
    print("Skipped", skipped, file=sys.stderr)
