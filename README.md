# gatbert

## Extracting a ConceptNet Subgraph

One way or another, you're going to need to have an instance of the ConceptNet postgres database running.
See our [guide](https://github.com/UF-NLPC-Lab/Guides/tree/main/conceptnet) on hosting ConceptNet using Apptainer.

All below commands assuming your instance is running on `127.0.0.1:5432`.
If not, they do accept a `-pg` CLI argument that lets you specify custom postgres connection parameters.

Once you have the instance running, you'll need to create some truncated tables that we use for our graph extraction:
```bash
conda activate gatbert
python -m gatbert.preproc_cndb --all
```

Further instructions pending...
