# STL
import dataclasses
from typing import List, Optional
from itertools import chain
# 3rd Party
import torch
from transformers import BartTokenizerFast, BartModel
from transformers.utils.generic import ModelOutput
# Local
from .data import Sample
from .types import TensorDict
from .encoder import keyed_scalar_stack, keyed_pad, Encoder, get_text_masks, SimpleEncoder
from .constants import EzstanceDomains, Stance
from .base_module import StanceModule


class ReconLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, predicted, original, mask):
        mask = torch.unsqueeze(mask, -1)
        # A bit inefficient because we use ReconLoss twice and recompute this activation function
        yhat = predicted * mask
        y = original * mask
        norms = torch.linalg.norm(yhat - y, dim=-1) ** 2
        loss = norms.mean()
        return loss

class AdvModule(StanceModule):

    training_only = ["domain", "context_mask", "target_mask"]

    @dataclasses.dataclass
    class Output(ModelOutput):
        logits: Optional[torch.FloatTensor] = None
        last_hidden_state: Optional[torch.FloatTensor] = None
        dom_logits: Optional[torch.FloatTensor] = None

    def __init__(self,
                 held_out: str,
                 pretrained_model: str = 'facebook/bart-large-mnli',
                 recon_weight: float = 0.0,
                 reg_weight: float = 0.0,
                 adv_weight: float = 0.0):
        super().__init__()
        domains = [dom.value for dom in EzstanceDomains] # TODO: let user specify what domains they're testing over

        bart_model = BartModel.from_pretrained(pretrained_model)
        tokenizer = BartTokenizerFast.from_pretrained(pretrained_model)
        config = bart_model.config
        hidden_size = config.d_model #FIXME
        self.bart_encoder = bart_model.encoder
        self.embeddings = bart_model.shared

        # The transformation which we regularize
        self.trans_layer = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        predictor_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.activation_dropout
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(predictor_dropout),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, len(Stance), bias=True)
        )
        self.domain_head = torch.nn.Sequential(
            torch.nn.Dropout(predictor_dropout),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, len(domains) - 1, bias=True)
        )

        self.recon_act = torch.nn.ReLU()
        self.recon_loss = ReconLoss()
        self.stance_loss = torch.nn.CrossEntropyLoss()
        self.adv_loss = torch.nn.CrossEntropyLoss()

        self.recon_weight = recon_weight
        self.reg_weight = reg_weight
        self.adv_weight = adv_weight
        self.__encoder = AdvModule.Encoder(held_out=held_out,
                                           domains=domains,
                                           tokenizer=tokenizer)
        self.register_buffer("identity_trans", torch.eye(hidden_size), persistent=False)

    @property
    def encoder(self):
        return self.__encoder

    def get_optimizer_params(self):
        """
        This class is the entire motivation for this get_optimizer_params method
        Want the wrapped module to be able to set its own learning rates
        """
        wrapped_params = chain(self.bart_encoder.parameters(), self.embeddings.parameters())
        new_params = chain(self.trans_layer.parameters(), self.classifier.parameters(), self.domain_head.parameters())
        return [
            {"params": wrapped_params, "lr": 2e-5},
            {"params": new_params, "lr": 1e-3}
        ]

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        input_ids = batch.pop('input_ids')
        context_mask = batch.pop('context_mask')
        target_mask = batch.pop('target_mask')
        inputs_embeds = self.embeddings(input_ids)

        domains = batch.pop('domain')
        adv_output = self(inputs_embeds=inputs_embeds, **batch)

        stance_loss_val = self.stance_loss(adv_output.logits, labels)
        adv_loss_val = self.adv_loss(adv_output.dom_logits, domains)
        trans_w = self.trans_layer.weight 
        reg_loss_val = torch.linalg.norm(trans_w - self.identity_trans)**2

        act_static_embeds = self.recon_act(inputs_embeds)
        act_context_embeds = self.recon_act(adv_output.last_hidden_state)
        recon_loss_val = self.recon_loss(act_context_embeds, act_static_embeds, context_mask) + \
            self.recon_loss(act_context_embeds, act_static_embeds, target_mask)

        loss = \
              self.recon_weight *  recon_loss_val \
            +   self.reg_weight *    reg_loss_val \
            -   self.adv_weight *    adv_loss_val \
            +                     stance_loss_val

        self.log('loss', loss)
        self.log('loss/domain', adv_loss_val)
        self.log('loss/recon', recon_loss_val)
        self.log('loss/reg', reg_loss_val)
        self.log('loss/stance', stance_loss_val)
        return loss

    def validation_step(self, batch, batch_idx):
        for k in AdvModule.training_only:
            batch.pop(k)
        return super().validation_step(batch, batch_idx)
    def test_step(self, batch, batch_idx):
        for k in AdvModule.training_only:
            batch.pop(k)
        return super().test_step(batch, batch_idx)

    def forward(self, **kwargs):
        base_output = self.bart_encoder(**kwargs)
        last_hidden_state = base_output.last_hidden_state
        transformed = self.trans_layer(last_hidden_state[:, 0])
        stance_logits = self.classifier(transformed)
        dom_logits = self.domain_head(transformed)
        return AdvModule.Output(
            logits=stance_logits,
            last_hidden_state=last_hidden_state,
            dom_logits=dom_logits
        )

    class Encoder(Encoder):
        def __init__(self,
                     held_out,
                     domains,
                     tokenizer):
            self.wrapped = SimpleEncoder(tokenizer)

            self.domain2id = {}
            i = -1
            for dom in domains:
                if dom != held_out:
                    self.domain2id[dom] = (i := i + 1)
            self.domain2id[held_out] = i + 1

        def encode(self, sample: Sample):
            encoding = self.wrapped.encode(sample)
            encoding['context_mask'], encoding['target_mask'] = get_text_masks(encoding['special_tokens_mask'])
            encoding['domain'] = torch.tensor(self.domain2id[sample.domain])
            return encoding

        def collate(self, samples: List[TensorDict]) -> TensorDict:
            batch = self.wrapped.collate(samples)
            batch['domain'] = keyed_scalar_stack(samples, 'domain')
            batch['target_mask'] = keyed_pad(samples, 'target_mask')
            batch['context_mask'] = keyed_pad(samples, 'context_mask')
            return batch