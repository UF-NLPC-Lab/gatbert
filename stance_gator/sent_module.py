from collections import namedtuple
import typing
# 3rd Party
import torch
from transformers import BertModel, BertTokenizerFast, PreTrainedTokenizerFast
# Local
from .sample import Sample
from .base_module import StanceModule
from .constants import DEFAULT_MODEL, TriStance
from .encoder import SimpleEncoder, Encoder, keyed_scalar_stack, collate_ids, keyed_pad

class SentModule(StanceModule):
    def __init__(self,
                pretrained_model = DEFAULT_MODEL,
                **parent_kwargs
                ):
        super().__init__(**parent_kwargs)
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
            torch.nn.Linear(feature_size, 1, bias=True),
            torch.nn.Tanh(),
            torch.nn.Flatten(start_dim=-2, end_dim=-1)
        )

        self.loss_func = torch.nn.MSELoss()

        self.__encoder = self.Encoder(BertTokenizerFast.from_pretrained(pretrained_model))

        self.att_scale: torch.Tensor
        self.register_buffer("att_scale", torch.sqrt(torch.tensor(self.hidden_size)), persistent=False)

    @property
    def encoder(self):
        return self.__encoder

    Output = namedtuple("SentOutput", field_names=["stance_vals", "loss"])

    def forward(self, context, target, context_mask, labels=None):
        context_output = self.bert(**context)
        context_hidden_states = context_output.last_hidden_state

        target_output = self.bert(**target)
        target_features = target_output.last_hidden_state[:, 0]

        attention_logits = torch.squeeze(torch.matmul(context_hidden_states, torch.unsqueeze(target_features, -1))) / self.att_scale
        attention_logits = attention_logits + torch.where(context_mask, 0, -torch.inf)
        attention_probs = torch.softmax(attention_logits, dim=-1)

        token_sent_vals = self.sent_classifier(context_hidden_states)
        seq_stance_vals = torch.sum(attention_probs * token_sent_vals, dim=-1)

        loss = None
        if labels is not None:
            labels = torch.where(labels == TriStance.favor, 1., torch.where(labels == TriStance.against, -1., 0.))
            loss = self.loss_func(seq_stance_vals, labels)
        return self.Output(seq_stance_vals, loss)

    def _eval_step(self, batch, batch_idx):
        labels = batch.pop('labels').view(-1)
        stance_vals, _ = self(**batch)

        preds = torch.where(stance_vals > .5, TriStance.favor, torch.where(stance_vals < .5, TriStance.against, TriStance.neutral))
        self._calc.record(preds, labels)


    class Encoder(Encoder):
        def __init__(self, tokenizer: PreTrainedTokenizerFast):
            self.tokenizer = tokenizer

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
            rdict = {
                "context": collate_ids(self.tokenizer, [s['context'] for s in samples], return_attention_mask=True),
                "target" : collate_ids(self.tokenizer, [s['target'] for s in samples],  return_attention_mask=True),
                "labels" : keyed_scalar_stack(samples, 'labels'),
                "context_mask": keyed_pad(samples, 'context_mask', False)
            }
            return rdict

