from typing import List
# 3rd Party
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast, PreTrainedTokenizerFast
# Local
from .sample import Sample, PretokenizedSample
from .encoder import encode_text, keyed_scalar_stack, collate_ids, Encoder
from .constants import DEFAULT_MODEL, Stance
from .base_module import StanceModule
from .types import TensorDict

class BertModule(StanceModule):
    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL,
                 predictor_bias: bool = False):
        super().__init__()

        label2id = {s.name:s.value for s in Stance}
        id2label = {v:k for k,v in label2id.items()}
        self.config = BertConfig.from_pretrained(pretrained_model, id2label=id2label, label2id=label2id)
        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model, config=self.config)
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(pretrained_model)

        if not predictor_bias:
            # The bert config doesn't let us enable or disable the bias, so we just set it to 0
            # and never let it get updated during training
            pred_bias = self.bert.classifier.bias
            pred_bias.data[:] = 0.
            pred_bias.requires_grad = False

        self.__encoder = self.Encoder(self.tokenizer)

    @property
    def encoder(self) -> Encoder:
        return self.__encoder

    def forward(self, *args, **kwargs):
        bert_output = self.bert(*args, **kwargs)
        return bert_output.logits

    class Encoder(Encoder):
        def __init__(self, tokenizer: PreTrainedTokenizerFast):
            self.__tokenizer = tokenizer
        def encode(self, sample: Sample | PretokenizedSample):
            return {
                **encode_text(self.__tokenizer, sample),
                'stance': torch.tensor([sample.stance.value])
            }
        def collate(self, samples: List[TensorDict]) -> TensorDict:
            return {
                **collate_ids(self.__tokenizer, samples, return_attention_mask=True),
                'stance': keyed_scalar_stack(samples, 'stance')
            }

