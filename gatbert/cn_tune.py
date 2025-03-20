# STL
import sys
import gc
import os
import pathlib
import argparse
# 3rd Party
import torch
from transformers import BertTokenizerFast, Trainer, TrainingArguments, BertForPreTraining, EarlyStoppingCallback
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
    parser.add_argument("-d", type=pathlib.Path, required=True, metavar="output/", help="Output directory containing the CN triples, the HF checkpoint dirs, and the final HF saved model")
    args = parser.parse_args()

    os.makedirs(args.d, exist_ok=True)

    if args.cn is not None:
        save_all_triples(ConceptNet(name=args.cn, create_inverse_triples=True), args.d)
        # Just to be safe; don't want that memory lingering since we've had memory issues
        # with saving model weights at the end
        gc.collect()

    # A bit inefficient to re-read the entities we just wrote to disk,
    # but good enough for now
    graph = CNGraph.from_pykeen(args.d)
    id2uri = graph.id2uri

    rng = np.random.default_rng(seed=0)

    node_ids = sorted(id2uri)
    desired_pos = len(node_ids)
    desired_neg = desired_pos

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
            # This loop has never actually run because our dataset has no islands (but yours might)
            head = islands[i]
            tail = rng.choice(non_islands)
            chosen_neg.add((head, tail))
            progress_bar.update()
            i += 1
        # The islands may not have been enough to get to our desired ratio
        n_nodes = len(node_ids)
        while len(chosen_neg) < desired_neg:
            # Get an edge we don't already have
            # Turns out rng.choice(n_nodes) is much quicker than rng.choice(node_ids)
            candidate = (rng.choice(n_nodes), rng.choice(n_nodes))
            while candidate in chosen_pos or candidate in chosen_neg:
                candidate = (rng.choice(node_ids), rng.choice(node_ids))
            chosen_neg.add(candidate)
            progress_bar.update()
        
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(args.pretrained)
    tokenizer.save_pretrained(args.d)
    id2text = {id : pretokenize_cn_uri(graph.id2uri[id]) for id in node_ids}
    def encode(head_id, tail_id, nsp_label):
        encoding = tokenizer(text=id2text[head_id],
                             text_pair=id2text[tail_id],
                             is_split_into_words=True,
                             return_special_tokens_mask=True,
                             # Setting 'pt' breaks collator?
                             return_tensors=None
                             )
        encoding['next_sentence_label'] = nsp_label
        return encoding

    # Just to ensure determinacy. Python hashmaps aren't random but implementations can vary
    sample_ids = [(head, tail, 0) for (head, tail) in sorted(chosen_pos)] + \
                [(head, tail, 1) for (head, tail) in sorted(chosen_neg)]

    # From the HF tutorial--still don't understand why
    ds = Dataset.from_list([encode(*sample) for sample in tqdm(sample_ids, desc="Tokenization")])
    splits = ds.train_test_split(seed=0)

    class CustomCollator(DataCollatorForLanguageModeling):
        def torch_mask_tokens(self, inputs, special_tokens_mask = None):
            new_inputs, new_labels = super().torch_mask_tokens(inputs, special_tokens_mask)
            # return new_inputs, new_labels
            # Always make sure at least one token is not set to -100
            # Otherwise we get NaN losses
            # We know the first token is CLS, so it's an easy guess
            # First token of the first sample
            index = tuple(0 for _ in new_inputs.shape)
            new_labels[index] = inputs[index]
            return new_inputs, new_labels

    trainer_args = TrainingArguments(
        output_dir=args.d,
        overwrite_output_dir=False,

        save_strategy="steps",
        save_steps=.01,

        eval_strategy='steps',
        eval_steps=.01,

        learning_rate=5e-5,
        num_train_epochs=50,

        metric_for_best_model='loss',
        load_best_model_at_end=True
    )

    model = BertForPreTraining.from_pretrained(args.pretrained)
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=splits['train'],
        eval_dataset=splits['test'],
        callbacks=[EarlyStoppingCallback(3)],
        data_collator=CustomCollator(tokenizer)
    )
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(args.d)
