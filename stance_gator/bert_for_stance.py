# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MODIFIED BY: Ethan Mines, 2025
# DESCRIPTION OF CHANGES: Modified the BertForSequenceClassification class for my own BertForStance class.

# STL
from typing import Optional
# 3rd Party
import torch
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertConfig, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.configuration_bert import BertConfig
# Local
from .output import StanceOutput

class BertForStanceConfig(BertConfig):
    def __init__(self,
                 *,
                 classifier_hidden_units: Optional[int] = None,
                 **base_kwargs):
        super().__init__(**base_kwargs)
        self.problem_type = "single_label_classification"
        self.add_pooling_layer = False
        self.return_dict = False
        self.classifier_hidden_units = classifier_hidden_units if classifier_hidden_units else self.hidden_size

class BertForStance(BertPreTrainedModel):
    def __init__(self, config: BertForStanceConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        hidden_size = config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob),
            torch.nn.Linear(hidden_size, config.classifier_hidden_units, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(config.classifier_hidden_units, self.num_labels, bias=True)
        )
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.post_init()

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
        return StanceOutput(loss=loss, logits=logits, seq_encoding=feature_vec)