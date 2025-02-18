# gatbert

## Downloading Data

- [EZStance](https://github.com/chenyez/EZ-STANCE)
- [VAST](https://github.com/emilyallaway/zero-shot-stance/tree/master/data/VAST)
- [SemEval2016-Task6](https://www.saifmohammad.com/WebDocs/stance-data-all-annotations.zip)


## ConceptNet Preprocessing

### Extracting a ConceptNet Sugraph

One way or another, you're going to need to have an instance of the ConceptNet postgres database running.
See our [guide](https://github.com/UF-NLPC-Lab/Guides/tree/main/conceptnet) on hosting ConceptNet using Apptainer.

All below commands assuming your instance is running on `127.0.0.1:5432`.
If not, they do accept a `-pg` CLI argument that lets you specify custom postgres connection parameters.

Once you have the instance running, you'll need to create some truncated tables that we use for our graph extraction:
```bash
conda activate gatbert
python -m gatbert.preproc_cndb --all
```
I'd budget 12 hours for this process (actually runtime will probably be less than that, but you need a buffer).

In our work, we only use training samples to obtain seed concepts from ConceptNet.
Here's an example of how to extract a subgraph:
```bash
conda activate gatbert
python -m gatbert.extract_cn --ezstance /path/to/ezstance/subtaskA/noun_phrase/raw_train_all_onecol.csv -o graph.json
```
I'd budget 4 hours for this process.

### Making ConceptNet Graph Samples

Then you can use the extracted graph to make graph-based samples (from both training and other partitions):
```bash
conda activate gatbert
python -m gatbert.tag --ezstance /path/to/ezstance/subtaskA/noun_phrase/raw_train_all_onecol.csv --graph graph.json -o train_graph.tsv
python -m gatbert.tag --ezstance /path/to/ezstance/subtaskA/noun_phrase/raw_val_all_onecol.csv   --graph graph.json -o val_graph.tsv
```
I'd budget an hour for this process.
