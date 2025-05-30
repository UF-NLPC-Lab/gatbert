from __future__ import annotations
from typing import Optional, List
import dataclasses
# 3rd Party
import torch
from transformers import BertModel, BertTokenizerFast, PreTrainedTokenizerFast
from transformers.utils.generic import ModelOutput
# Local
from .sample import Sample
from .base_module import StanceModule
from .constants import DEFAULT_MODEL, TriStance
from .encoder import Encoder, keyed_scalar_stack, collate_ids, keyed_pad

class SentModule(StanceModule):
    def __init__(self,
                pretrained_model = DEFAULT_MODEL,
                **parent_kwargs
                ):
        super().__init__(**parent_kwargs)
        self.save_hyperparameters()
        assert self.stance_enum == TriStance

        self.bert = BertModel.from_pretrained(pretrained_model)
        config = self.bert.config

        hidden_size = config.hidden_size
        feature_size = hidden_size
        self.hidden_size = hidden_size

        self.sent_classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob),
            torch.nn.Linear(feature_size, feature_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_size, len(self.stance_enum), bias=True),
            torch.nn.Softmax(dim=-1)
        )

        self.loss_func = torch.nn.MSELoss()

        self.__encoder = self.Encoder(BertTokenizerFast.from_pretrained(pretrained_model))

        self.att_scale: torch.Tensor
        self.register_buffer("att_scale", torch.sqrt(torch.tensor(self.hidden_size)), persistent=False)

    @property
    def encoder(self) -> SentModule.Encoder:
        return self.__encoder

    @dataclasses.dataclass
    class Output(ModelOutput):
        token_sents: torch.Tensor
        stance_prob: Optional[torch.Tensor] = None
        attention: Optional[torch.Tensor] = None
        loss: Optional[torch.Tensor] = None
        hidden_states: Optional[torch.Tensor] = None


    def predict_sent(self, context, context_mask):
        context_output = self.bert(**context)
        context_hidden_states = context_output.last_hidden_state
        token_sent_vals = self.sent_classifier(context_hidden_states)

        masked_states = token_sent_vals * torch.unsqueeze(context_mask, dim=-1)
        summed = torch.sum(masked_states, dim=1)
        n_tokens = torch.sum(context_mask, dim=-1, keepdim=True)
        prob_dist = summed / n_tokens
        return prob_dist

    def forward(self, context, target=None, context_mask=None, labels=None, return_hidden_states=False):
        context_output = self.bert(**context)
        context_hidden_states = context_output.last_hidden_state
        token_sents = self.sent_classifier(context_hidden_states)

        loss = None
        attention = None
        stance_prob = None
        if target is not None and context_mask is not None:
            target_output = self.bert(**target)
            target_features = target_output.last_hidden_state[:, 0]
            attention_logits = torch.squeeze(torch.matmul(context_hidden_states, torch.unsqueeze(target_features, -1))) / self.att_scale
            attention_logits = attention_logits + torch.where(context_mask, 0, -torch.inf)
            attention = torch.softmax(attention_logits, dim=-1)

            stance_prob = torch.sum(torch.unsqueeze(attention, dim=-1) * token_sents, dim=-2)
            if labels is not None:
                labels = torch.nn.functional.one_hot(labels, num_classes=len(self.stance_enum)).to(torch.float)
                loss = self.loss_func(stance_prob, labels)
        elif context_mask is not None:
            # Compute a uniform average over the sentiment
            masked_states = token_sents * torch.unsqueeze(context_mask, dim=-1)
            summed = torch.sum(masked_states, dim=1)
            n_tokens = torch.sum(context_mask, dim=-1, keepdim=True)
            stance_prob = summed / n_tokens

        return SentModule.Output(token_sents=token_sents,
                                 stance_prob=stance_prob,
                                 attention=attention,
                                 loss=loss,
                                 hidden_states=context_hidden_states if return_hidden_states else None)

    def _eval_step(self, batch, batch_idx):
        labels = batch.pop('labels').view(-1)
        res = self(**batch)
        preds = res.stance_prob
        self._calc.record(preds, labels)


    class Encoder(Encoder):
        def __init__(self, tokenizer: PreTrainedTokenizerFast):
            self.tokenizer = tokenizer

        def encode_sentiment(self, text: List[str], label: int):
            encoding = self.tokenizer(text=text, return_special_tokens_mask=True, return_tensors='pt')
            encoding['token_type_ids'] = torch.ones_like(encoding['input_ids'])
            special_tokens_mask = encoding.pop('special_tokens_mask')
            context_mask = torch.logical_not(special_tokens_mask)
            return {
                "context": encoding,
                "context_mask": context_mask,
                "labels": torch.tensor(label)
            }

        def encode(self, sample: Sample):

            tokenizer_kwargs = {
                'is_split_into_words': sample.is_split_into_words,
                "return_token_type_ids": False,
                "return_tensors": "pt"
            }

            target_encoding  = self.tokenizer(text=sample.target, **tokenizer_kwargs)
            target_encoding['token_type_ids'] = torch.zeros_like(target_encoding['input_ids'])
            context_encoding = self.tokenizer(text=sample.context, return_special_tokens_mask=True, **tokenizer_kwargs)
            special_tokens_mask = context_encoding.pop('special_tokens_mask')
            context_mask = torch.logical_not(special_tokens_mask)
            context_encoding['token_type_ids'] = torch.ones_like(context_encoding['input_ids'])
            return {
                "context": context_encoding,
                "target": target_encoding,
                "labels": torch.tensor(int(sample.stance)),
                "context_mask": context_mask
            }

        
        def collate(self, samples):
            rdict = {}
            rdict['context'] = collate_ids(self.tokenizer, [s['context'] for s in samples], return_attention_mask=True)
            rdict['labels'] = keyed_scalar_stack(samples, 'labels')
            rdict['context_mask'] = keyed_pad(samples, 'context_mask', False)
            if "target" in samples[0]:
                rdict['target'] = collate_ids(self.tokenizer, [s['target'] for s in samples],  return_attention_mask=True)
            return rdict

