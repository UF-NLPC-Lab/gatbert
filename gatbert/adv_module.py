# STL
from typing import List
import math
# 3rd Party
import torch
# Local
from .data import Sample, PretokenizedSample
from .types import TensorDict
from .encoder import keyed_scalar_stack, SimpleEncoder, Encoder
from .constants import DEFAULT_MODEL, EzstanceDomains
from .base_module import StanceModule
from .bert_module import BertModule

class AdvModule(StanceModule):

    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL,
                 gamma: float = 10):
        super().__init__()

        self.bert_mod = BertModule(pretrained_model)
        self.hidden_size = 128

        self.domain_head = torch.nn.Sequential(
            torch.nn.Dropout(p=0.20),
            torch.nn.Linear(2 * self.bert_mod.bert.config.hidden_size, self.hidden_size, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_size, len(EzstanceDomains), bias=True)
        )

        self.stance_loss = torch.nn.CrossEntropyLoss()
        self.dom_loss = torch.nn.CrossEntropyLoss()

        self.gamma = gamma
        self.adv_weight = 0
        self.epoch_i = -1
        self.max_adv_epochs = 100
        self.__encoder = AdvModule.Encoder(self.bert_mod.encoder)

    @property
    def encoder(self):
        return self.__encoder

    def on_train_epoch_start(self):
        self.epoch_i += 1
        self.adv_weight = max(0,
            2 / (1 + math.exp(-self.gamma * (self.epoch_i + 1) / self.max_adv_epochs)) - 1
        )

    def training_step(self, batch, batch_idx):
        labels = batch.pop('stance')
        domains = batch.pop('domain')
        (stance_logits, dom_logits) = self(**batch)

        stance_loss_val = self.stance_loss(stance_logits, labels)
        dom_loss_val = self.dom_loss(dom_logits, domains)
        loss = stance_loss_val - self.adv_weight * dom_loss_val
        self.log('loss', loss)
        self.log('loss/domain', dom_loss_val)
        self.log('loss/stance', stance_loss_val)
        return loss

    def validation_step(self, batch, batch_idx):
        batch.pop('domain')
        return super().validation_step(batch, batch_idx)
    def test_step(self, batch, batch_idx):
        batch.pop('domain')
        return super().test_step(batch, batch_idx)

    def forward(self, **kwargs):
        (logits, context_vec, target_vec) = self.bert_mod(**kwargs)
        feature_vec = torch.concatenate([context_vec, target_vec], dim=-1)
        dom_logits = self.domain_head(feature_vec)
        return logits, dom_logits

    class Encoder(Encoder):
        def __init__(self, wrapped: SimpleEncoder):
            self.wrapped = wrapped
            self.domain2id = {dom:i for i,dom in enumerate(EzstanceDomains)}

        def encode(self, sample: Sample | PretokenizedSample):
            encoding = self.wrapped.encode(sample)
            encoding['domain'] = torch.tensor(self.domain2id[sample.domain])
            return encoding

        def collate(self, samples: List[TensorDict]) -> TensorDict:
            domains = keyed_scalar_stack(samples, 'domain')
            batch = self.wrapped.collate(samples)
            batch['domain'] = domains
            return batch