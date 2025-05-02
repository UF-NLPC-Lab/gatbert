# STL
from typing import List
import math
# 3rd Party
import torch
# Local
from .data import Sample, PretokenizedSample
from .types import TensorDict
from .encoder import keyed_scalar_stack, Encoder
from .constants import EzstanceDomains
from .base_module import StanceModule

class AdvModule(StanceModule):

    def __init__(self,
                 held_out: EzstanceDomains,
                 wrapped: StanceModule,
                 dropout: float = 0.0,
                 gamma: float = 10):
        super().__init__()

        self.held_out = held_out
        self.wrapped = wrapped

        self.domain_head = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(self.feature_size, self.feature_size, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(self.feature_size, len(EzstanceDomains) - 1, bias=True)
        )

        self.stance_loss = torch.nn.CrossEntropyLoss()
        self.dom_loss = torch.nn.CrossEntropyLoss()

        self.gamma = gamma
        self.adv_weight = 0
        self.epoch_i = -1
        self.max_adv_epochs = 100
        self.__encoder = AdvModule.Encoder(held_out=self.held_out,
                                           domains=EzstanceDomains,
                                           wrapped=self.wrapped.encoder)

    @property
    def encoder(self):
        return self.__encoder

    @property
    def feature_size(self):
        return self.wrapped.feature_size

    def get_optimizer_params(self):
        """
        This class is the entire motivation for this get_optimizer_params method
        Want the wrapped module to be able to set its own learning rates
        """
        wrapped_params = self.wrapped.get_optimizer_params()
        return wrapped_params + [{"params": self.domain_head.parameters(), "lr": 1e-3}]

    def on_train_epoch_start(self):
        self.epoch_i += 1
        self.adv_weight = max(0,
            2 / (1 + math.exp(-self.gamma * (self.epoch_i + 1) / self.max_adv_epochs)) - 1
        )

    def training_step(self, batch, batch_idx):
        labels = batch.pop('stance')
        domains = batch.pop('domain')
        (stance_logits, _, dom_logits) = self(**batch)

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
        (logits, feature_vec) = self.wrapped(**kwargs)[:2]
        dom_logits = self.domain_head(feature_vec)
        return logits, feature_vec, dom_logits

    class Encoder(Encoder):
        def __init__(self,
                     held_out,
                     domains,
                     wrapped: Encoder):
            self.wrapped = wrapped

            self.domain2id = {}
            i = -1
            for dom in domains:
                if dom != held_out:
                    self.domain2id[dom] = (i := i + 1)
            self.domain2id[held_out] = i + 1

        def encode(self, sample: Sample | PretokenizedSample):
            encoding = self.wrapped.encode(sample)
            encoding['domain'] = torch.tensor(self.domain2id[sample.domain])
            return encoding

        def collate(self, samples: List[TensorDict]) -> TensorDict:
            domains = keyed_scalar_stack(samples, 'domain')
            batch = self.wrapped.collate(samples)
            batch['domain'] = domains
            return batch