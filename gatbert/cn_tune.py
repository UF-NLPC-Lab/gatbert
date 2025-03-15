# STL
import os
import pathlib
import argparse
# 3rd Party
from transformers import BertTokenizerFast, Trainer, TrainingArguments, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from tqdm import tqdm
from pykeen.datasets import ConceptNet
# Local
from .constants import DEFAULT_MODEL
from .pykeen_utils import save_all_triples
from .graph import CNGraph
from .encoder import pretokenize_cn_uri


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
    uris = [uri for uri, _ in sorted(uri2id.items(), key = lambda pair: pair[1])]

    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(args.pretrained)
    tokenizer.save_pretrained(args.d)

    # From the HF tutorial--still don't understand why
    ds = Dataset.from_list([tokenizer(text=pretokenize_cn_uri(uri), is_split_into_words=True, return_special_tokens_mask=True) for uri in tqdm(uris)])
    # splits = ds.train_test_split(seed=0)

    trainer_args = TrainingArguments(
        save_strategy="steps",
        save_steps=.25,
        learning_rate=1e-5,
        num_train_epochs=1,
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