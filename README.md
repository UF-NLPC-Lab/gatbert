# gatbert

## Dependencies

```bash
conda env create -f environment.yaml
conda activate gatbert
```

## Downloading Data

Stance Datasets:
- [EZStance](https://github.com/chenyez/EZ-STANCE)
- [VAST](https://github.com/emilyallaway/zero-shot-stance/tree/master/data/VAST)
- [SemEval2016-Task6](https://www.saifmohammad.com/WebDocs/stance-data-all-annotations.zip)

Knowledge Graph Datasets:
- [Concept Net Assertions](https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz)

## Graph Preprocessing

```bash
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gunzip conceptnet-assertions-5.7.0.csv.gz
utils/cn_grep.sh < conceptnet-assertions-5.7.0.csv > conceptnet-assertions-5.7.0-en.csv
```

Our [tag](gatbert/tag.py) module expects these assertions to be in the `.tsv.gz` triples format used by [PyKeen](https://pykeen.readthedocs.io/en/stable/).

Our general workflow is to use PyKeen to create the triples files and the node embeddings in one script run:
```bash
python -m gatbert.embed_kb \
	-cn conceptnet-assertions-5.7.0-en.csv \
	--embed TransE \
	-o /conceptnet/data
```

Alternatively, if you only want the triples you can omit the `--embed` argument:
```bash
python -m gatbert.embed_kb \
	-cn conceptnet-assertions-5.7.0-en.csv \
	-o /conceptnet/data
```

### Making ConceptNet Graph Samples

Then you can use the extracted graph to make graph-based samples (from both training and other partitions):
```bash
conda activate gatbert
python -m gatbert.tag --ezstance /path/to/ezstance/subtaskA/noun_phrase/raw_train_all_onecol.csv --graph /conceptnet/data -o train_graph.tsv
python -m gatbert.tag --ezstance /path/to/ezstance/subtaskA/noun_phrase/raw_val_all_onecol.csv   --graph /conceptnet/data -o val_graph.tsv
```
I'd budget four hours for this process (some datasets are larger than others.)


## Running Experiments

If you're a developer on this project, you should read our [PyTorch Lightning guide](https://github.com/UF-NLPC-Lab/Guides/tree/main/pytorch-lightning),
as it provides additional context on our development environment.

Here's an example of how to run different models with different data using PyTorch lightning configs:

```bash
function run_exp()
{
	python -m gatbert.fit_and_test $@ --print_config
	python -m gatbert.fit_and_test $@
}

run_exp \
	-c sample_configs/base.yaml \
	--data sample_configs/graph_data.yaml \
	--model.classifier gatbert.stance_classifier.BertClassifier \
	--trainer.logger.init_args.version bert_$(date +%s)

run_exp \
	-c sample_configs/base.yaml \
	--data sample_configs/tagged_data.yaml \
	--model.classifier gatbert.stance_classifier.ConcatClassifier \
	--model.classifier.graph /conceptnet/data \
	--model.classifier.graph_model cgcn \
	--trainer.logger.init_args.version two_model_cgcn_$(date +%s)

run_exp \
	-c sample_configs/base.yaml \
	--data sample_configs/tagged_data.yaml \
	--model.classifier gatbert.stance_classifier.ConcatClassifier \
	--model.classifier.graph /conceptnet/data \
	--model.classifier.graph_model gat \
	--trainer.logger.init_args.version two_model_gat_$(date +%s)

run_exp \
	-c sample_configs/base.yaml \
	--data sample_configs/tagged_data.yaml \
	--model.classifier gatbert.stance_classifier.HybridClassifier \
	--model.classifier.graph /conceptnet/data \
	--trainer.logger.init_args.version hybrid_$(date +%s)
```

You will need to update the following fields to run it on your machine:
- `base.yaml`: `trainer.logger.init_args.save_dir`
- `raw_data.yaml`: the data paths under `init_args.partitions`
- `graph_data.yaml`: the data paths under `init_args.partitions`
