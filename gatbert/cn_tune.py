# STL
import sys
import os
import pathlib
import argparse
# 3rd Party
from transformers import BertTokenizerFast, Trainer, TrainingArguments, BertForPreTraining
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from tqdm import tqdm
from pykeen.datasets import ConceptNet
import numpy as np
# Local
from .constants import DEFAULT_MODEL
from .pykeen_utils import save_all_triples
from .graph import CNGraph
from .encoder import pretokenize_cn_uri
from .utils import time_block


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-cn", metavar="conceptnet-assertions-5.7.0.csv", type=str,
                        help="Path to conceptnet assertions")
    parser.add_argument("--pretrained", default=DEFAULT_MODEL, metavar=DEFAULT_MODEL, help="Pretrained model to intialize BERT")
    parser.add_argument("-d", type=pathlib.Path, required=True, metavar="output/", help="Output directory containing the CN triples, and the HuggingFace saved model")
    args = parser.parse_args()

    os.makedirs(args.d, exist_ok=True)

    if args.cn is not None:
        pykeen_ds = ConceptNet(name=args.cn, create_inverse_triples=True)
        save_all_triples(pykeen_ds, args.d)
        del pykeen_ds

    # A bit inefficient to re-read the entities we just wrote to disk,
    # but good enough for now
    graph = CNGraph.from_pykeen(args.d)

    uri2id = graph.uri2id
    id2uri = graph.id2uri

    rng = np.random.default_rng(seed=0)

    # Construct two samples per-entity
    node_ids = sorted(id2uri.keys())
    # Limiting this for debugging
    node_ids = node_ids[:5000]

    desired_pos = len(node_ids)
    desired_neg = len(node_ids)

    with time_block("all_pos"):
        all_pos = {(head, tail) for (head, edges) in graph.adj.items() for (tail, _) in edges}
    if len(all_pos) < desired_pos:
        print(f"Not enough edges to meet quota. Reducing number of positive samples to {len(all_pos)}")
        desired_pos = len(all_pos)

    with tqdm(total=desired_pos, desc="Positive NSP Sample Selection") as pos_progress_bar:
        chosen_pos = set()
        non_islands = sorted(n for n in node_ids if n in graph.adj)
        # Round robin allocation of edges from each node
        while len(chosen_pos) < desired_pos:
            for node_id in non_islands:
                edges = graph.adj[node_id]
                if edges:
                    # Take a random edge as a sample
                    index = rng.choice(len(edges))
                    # And remove it from the graph so we don't use it later
                    edge = edges.pop(index)
                    (tail, _) = edge

                    chosen_pos.add( (node_id, tail) )
                    pos_progress_bar.update()
                    if len(chosen_pos) >= desired_pos:
                        break

    with tqdm(total=desired_neg, desc="Negative NSP Sample Selection") as progress_bar:
        chosen_neg = set()
        # Node with no edges (recall we add inverse edges, so 0-outdegree also means 0-indegree)
        islands = [node_id for node_id in node_ids if node_id not in graph.adj]
        i = 0
        while i < len(islands) and len(chosen_neg) < desired_neg:
            head = islands[i]
            tail = rng.choice(non_islands)
            chosen_neg.add((head, tail))
            progress_bar.update()
        # The islands may not have been enough to get to our desired ratio
        while len(chosen_neg) < desired_neg:
            # Get an edge we don't already have
            candidate = (rng.choice(node_ids), rng.choice(node_ids))
            while candidate in chosen_pos or candidate in chosen_neg:
                candidate = (rng.choice(node_ids), rng.choice(node_ids))
            chosen_neg.add(candidate)
            progress_bar.update()
        
    sys.exit(0)


    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(args.pretrained)
    tokenizer.save_pretrained(args.d)

    # From the HF tutorial--still don't understand why
    ds = Dataset.from_list([tokenizer(text=pretokenize_cn_uri(uri), is_split_into_words=True, return_special_tokens_mask=True) for uri in tqdm(uris)])
    # splits = ds.train_test_split(seed=0)

    trainer_args = TrainingArguments(
        save_strategy="steps",
        save_steps=.25,
        learning_rate=5e-5,
        num_train_epochs=4,
    )

    # If we have a batch where no token is masked,
    # then the loss will be nan.
    # So we set the mlm_probability to 1,
    # but only perform actual masking quite rarely
    mlm_probability = .15 #1.
    collator = DataCollatorForLanguageModeling(
        tokenizer,
        # mlm_probability=mlm_probability,
        # mask_replace_prob=.8*.15/mlm_probability,
        # random_replace_prob=.1*.15/mlm_probability
    )

    model = BertForMaskedLM.from_pretrained(args.pretrained)
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=ds,
        # eval_dataset=splits['test'],
        # callbacks=[EarlyStoppingCallback(3)],
        data_collator=collator
    )
    trainer.train()
    trainer.save_model(args.d)
