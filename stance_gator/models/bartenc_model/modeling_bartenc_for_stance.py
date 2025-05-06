import math
from typing import Optional
import dataclasses
# 3rd Party
import torch
from transformers import BartTokenizerFast
from transformers.models.bart.modeling_bart import BartEncoder, BartPreTrainedModel, BartScaledWordEmbedding
from transformers.utils.generic import ModelOutput
# Local
from .configuration_bartenc_for_stance import BartEncForStanceConfig


class BartEncForStance(BartPreTrainedModel):
    config_class = BartEncForStanceConfig
    _tied_weights_keys = ["encoder.embed_tokens.weight"]

    def __init__(self, config: BartEncForStanceConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        hidden_size = config.d_model

        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.shared = BartScaledWordEmbedding(config.vocab_size, config.d_model, config.pad_token_id, embed_scale=embed_scale)

        self.encoder = BartEncoder(config, self.shared)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.classifier_dropout if config.classifier_dropout is not None else config.activation_dropout),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.num_labels, bias=True)
        )
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.post_init()

    def _tie_weights(self):
        # FIXME: Address other niche cases liek "facebook/bart-large-cnn" like HF library did, if necessary
        self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    @dataclasses.dataclass
    class Output(ModelOutput):
        loss: Optional[torch.FloatTensor] = None
        logits: Optional[torch.FloatTensor] = None
        seq_encoding: Optional[torch.FloatTensor] = None

    def forward(self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.Tensor] = None):

        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True
        )
        feature_vec = encoder_out.last_hidden_state[:, 0]
        logits = self.classifier(feature_vec)
        loss = None
        if labels is not None:
           loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return BartEncForStance.Output(loss=loss, logits=logits, seq_encoding=feature_vec)

BartEncForStance.register_for_auto_class("AutoModel")