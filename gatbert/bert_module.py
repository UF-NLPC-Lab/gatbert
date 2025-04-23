from typing import List
# 3rd Party
import torch
from transformers import BertTokenizerFast, PreTrainedTokenizerFast, BertModel
# Local
from .sample import Sample, PretokenizedSample
from .encoder import encode_text, keyed_scalar_stack, collate_ids, Encoder, get_text_masks, keyed_pad
from .constants import DEFAULT_MODEL, Stance
from .base_module import StanceModule
from .types import TensorDict

class BertModule(StanceModule):
    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL):
        super().__init__()

        label2id = {s.name:s.value for s in Stance}
        id2label = {v:k for k,v in label2id.items()}
        hidden_size = 283 # Alloway's hparam
        # self.config = BertConfig.from_pretrained(pretrained_model, id2label=id2label, label2id=label2id)
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.ff = torch.nn.Sequential(
            torch.nn.Dropout(p=0.20463604390811982),
            torch.nn.Linear(2 * self.bert.config.hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, len(Stance), bias=True)
        )
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(pretrained_model)
        self.__encoder = self.Encoder(self.tokenizer, max_context_length=200, max_target_length=5)

    @property
    def encoder(self) -> Encoder:
        return self.__encoder

    @staticmethod
    def masked_average(mask, embeddings) -> torch.Tensor:
        # Need the epsilon to prevent divide-by-zero errors
        denom = torch.sum(mask, dim=-1, keepdim=True) + 1e-6
        return torch.sum(torch.unsqueeze(mask, -1) * embeddings, dim=-2) / denom

    def forward(self, **kwargs):
        target_text_mask = kwargs.pop('target_text_mask')
        context_text_mask = kwargs.pop('context_text_mask')
        # (1) Encode text
        bert_out = self.bert(**kwargs)
        hidden_states = bert_out.last_hidden_state
        target_text_vec = self.masked_average(target_text_mask, hidden_states)
        context_text_vec = self.masked_average(context_text_mask, hidden_states)
        feature_vec = torch.concatenate([target_text_vec, context_text_vec], dim=-1)
        logits = self.ff(feature_vec)
        return (logits, context_text_vec, target_text_vec)

    class Encoder(Encoder):
        def __init__(self, tokenizer: PreTrainedTokenizerFast, max_context_length: int, max_target_length: int):
            self.__tokenizer = tokenizer
            self.__max_context_length = max_context_length
            self.__max_target_length = max_target_length
        def encode(self, sample: Sample | PretokenizedSample):
            text_encoding = encode_text(self.__tokenizer, sample, self.__max_context_length, self.__max_target_length)
            target_text_mask, context_text_mask = get_text_masks(text_encoding.pop('special_tokens_mask'))

            return {
                **text_encoding,
                "target_text_mask": target_text_mask,
                "context_text_mask": context_text_mask,
                'stance': torch.tensor([sample.stance.value])
            }
        def collate(self, samples: List[TensorDict]) -> TensorDict:
            return {
                **collate_ids(self.__tokenizer, samples, return_attention_mask=True),
                'target_text_mask': keyed_pad(samples, 'target_text_mask'),
                'context_text_mask': keyed_pad(samples, 'context_text_mask'),
                'stance': keyed_scalar_stack(samples, 'stance')
            }

