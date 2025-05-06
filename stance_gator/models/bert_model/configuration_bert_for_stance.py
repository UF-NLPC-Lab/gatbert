from transformers.models.bert.configuration_bert import BertConfig
from typing import Optional

class BertForStanceConfig(BertConfig):
    model_type = "bert_for_stance"
    def __init__(self,
                 *,
                 classifier_hidden_units: Optional[int] = None,
                 **base_kwargs):
        super().__init__(**base_kwargs)
        self.problem_type = "single_label_classification"
        self.add_pooling_layer = False
        self.return_dict = True
        self.classifier_hidden_units = classifier_hidden_units if classifier_hidden_units else self.hidden_size
