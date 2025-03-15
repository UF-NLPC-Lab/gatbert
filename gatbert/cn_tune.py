# STL
import os
import pathlib
import argparse
from itertools import starmap, islice
# 3rd Party
import torch
from transformers import BertTokenizerFast, Trainer, TrainingArguments, BertModel, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import Dataset
from tqdm import tqdm
from pykeen.datasets import ConceptNet
# Local
from .utils import time_block
from .constants import DEFAULT_MODEL
from .pykeen_utils import save_all_triples
from .graph import CNGraph


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
    uri2id = CNGraph.read_entitites(args.d)
    uri2id = sorted(uri2id.items(), key = lambda pair: pair[1])

    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(args.pretrained)
    tokenizer.save_pretrained(args.d)

    samples = starmap(lambda uri, _: tokenizer(text=uri.split('_'), is_split_into_words=True, return_special_tokens_mask=True), tqdm(uri2id))
    samples = islice(samples, 256)
    samples = list(samples)
    ds = Dataset.from_list(samples)
    splits = ds.train_test_split(seed=0)

    trainer_args = TrainingArguments(
        eval_strategy="epoch",
        save_strategy="epoch",

        # eval_strategy='steps',
        # eval_steps=5,
        # logging_strategy='steps',
        # logging_steps=5,

        learning_rate=1e-5,
        num_train_epochs=1, #100,
        metric_for_best_model='loss',
        load_best_model_at_end=True
    )
    model = BertForMaskedLM.from_pretrained(args.pretrained)
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=splits['train'],
        eval_dataset=splits['test'],
        callbacks=[EarlyStoppingCallback(3)],
        data_collator=DataCollatorForLanguageModeling(tokenizer)
    )
    trainer.train()
    trainer.save_model(args.d)