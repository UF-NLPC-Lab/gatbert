import argparse
# 3rd Party
import lightning as L
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
# Local
from .torch_utils import load_module
from .constants import DEFAULT_BATCH_SIZE
from .sent_module import SentModule
from .constants import TriStance
from .data import MapDataset

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args(raw_args)

    sent_mod = load_module(args.ckpt)
    assert isinstance(sent_mod, SentModule)
    assert sent_mod.stance_enum is TriStance
    sent_mod.eval()

    hf_dataset = load_dataset("SetFit/sst5")
    label_map = {"very positive": TriStance.favor,
                 "positive": TriStance.favor,
                 "neutral": TriStance.neutral,
                 "negative": TriStance.against,
                 "very negative": TriStance.against}
    encoded = []
    for sample in hf_dataset['test']:
        encoded.append(sent_mod.encoder.encode_sentiment(sample['text'], label_map[sample['label_text']]))
    sent_dataset = MapDataset(encoded)
    sent_dataloader = DataLoader(sent_dataset,
                                 batch_size=DEFAULT_BATCH_SIZE,
                                 shuffle=False,
                                 collate_fn=sent_mod.encoder.collate)
    trainer = L.Trainer(logger=False, deterministic=True)
    trainer.test(sent_mod, sent_dataloader)

if __name__ == "__main__":
    main()