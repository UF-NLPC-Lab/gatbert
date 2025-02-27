# Utils

Simple utilities that don't require the conda environment.

## Parsing Metrics Files

```bash
./parse_met.py $(ls -d lightning_logs/*)
```

## Trimming non-English Data from a ConceptNet Assertions File

```bash
./cn_grep.sh < conceptnet-assertions-5.7.0.csv
```
