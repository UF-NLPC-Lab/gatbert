# STL
from typing import Optional
import dataclasses
# 3rd Party
import torch
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils.generic import ModelOutput
# Local
from .configuration_bert_for_stance import BertForStanceConfig


class BertForStance(BertPreTrainedModel):
    config_class = BertForStanceConfig

    def __init__(self, config: BertForStanceConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        hidden_size = config.hidden_size
        classifier_hidden_units = config.classifier_hidden_units or config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob),
            torch.nn.Linear(hidden_size, classifier_hidden_units, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(classifier_hidden_units, self.num_labels, bias=True)
        )
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.post_init()

    @dataclasses.dataclass
    class Output(ModelOutput):
        loss: Optional[torch.FloatTensor] = None
        logits: Optional[torch.FloatTensor] = None
        seq_encoding: Optional[torch.FloatTensor] = None
        last_hidden_state: Optional[torch.Tensor] = None


    def forward(
         self,
         input_ids: Optional[torch.Tensor] = None,
         attention_mask: Optional[torch.Tensor] = None,
         token_type_ids: Optional[torch.Tensor] = None,
         position_ids: Optional[torch.Tensor] = None,
         head_mask: Optional[torch.Tensor] = None,
         inputs_embeds: Optional[torch.Tensor] = None,
         labels: Optional[torch.Tensor] = None,
         return_dict: Optional[bool] = None,
     ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        feature_vec = outputs.last_hidden_state[:, 0]
        logits = self.classifier(feature_vec)
        loss = None
        if labels is not None:
           loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return BertForStance.Output(loss=loss, logits=logits, seq_encoding=feature_vec,
                                    last_hidden_state=outputs.last_hidden_state)

BertForStance.register_for_auto_class("AutoModel")