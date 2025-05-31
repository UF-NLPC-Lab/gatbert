import argparse
# 3rd Party
import lightning as L
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
# Local
from .torch_utils import load_module
from .constants import DEFAULT_BATCH_SIZE
from .base_module import StanceModule
from .sent_module import SentModule
from .constants import TriStance
from .data import MapDataset
from lightning.pytorch.loggers import CSVLogger

class FTModule(StanceModule):
    def __init__(self, sent_mod: SentModule):
        super().__init__()
        self.sent_mod = sent_mod
        for param in sent_mod.parameters():
            param.requires_grad_ = False
        self.query_vec = torch.nn.Parameter(torch.empty(self.sent_mod.hidden_size))
        torch.nn.init.uniform_(self.query_vec)

    @property
    def encoder(self) -> SentModule.Encoder:
        return self.sent_mod.encoder

    def forward(self, context, context_mask, labels=None):
        sent_output: SentModule.Output = self.sent_mod(context=context, context_mask=context_mask, return_hidden_states=True)
        token_sents = sent_output.token_sents
        context_hidden_states = sent_output.hidden_states
        attention_logits = torch.matmul(context_hidden_states, self.query_vec)
        attention_logits = attention_logits / self.sent_mod.att_scale
        attention_logits = attention_logits + torch.where(context_mask, 0, -torch.inf)
        attention = torch.softmax(attention_logits, dim=-1)

        stance_prob = torch.sum(torch.unsqueeze(attention, dim=-1) * token_sents, dim=-2)
        if labels is not None:
            labels = torch.nn.functional.one_hot(labels, num_classes=len(self.stance_enum)).to(torch.float)
            loss = self.sent_mod.loss_func(stance_prob, labels)
            sent_output.loss = loss
        sent_output.stance_prob = stance_prob
        return sent_output

    def _eval_step(self, batch, batch_idx):
        return SentModule._eval_step(self, batch, batch_idx)

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args(raw_args)

    orig_mod = load_module(args.ckpt)
    assert isinstance(orig_mod, SentModule)
    assert orig_mod.stance_enum is TriStance
    sent_mod = FTModule(orig_mod)

    hf_dataset = load_dataset("SetFit/sst5")
    label_map = {"very positive": TriStance.favor,
                 "positive": TriStance.favor,
                 "neutral": TriStance.neutral,
                 "negative": TriStance.against,
                 "very negative": TriStance.against}

    def make_dataset(subset):
        return MapDataset([
            sent_mod.encoder.encode_sentiment(sample['text'], label_map[sample['label_text']])
            for sample in subset
        ])
    
    logger = CSVLogger(save_dir='./')
    trainer = L.Trainer(logger=logger, deterministic=True, max_epochs=5)

    trainer.fit(sent_mod, DataLoader(make_dataset(hf_dataset['train']),
                                 batch_size=DEFAULT_BATCH_SIZE,
                                 shuffle=True,
                                 collate_fn=sent_mod.encoder.collate))

    trainer.test(sent_mod, DataLoader(make_dataset(hf_dataset['validation']),
                                 batch_size=DEFAULT_BATCH_SIZE,
                                 shuffle=False,
                                 collate_fn=sent_mod.encoder.collate))

if __name__ == "__main__":
    main()