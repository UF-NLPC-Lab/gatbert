from typing import Optional

from transformers.models.bart.configuration_bart import BartConfig

class BartEncForStanceConfig(BartConfig):
    model_type = "bart_enc_for_stance"

    def __init__(self,
                 *,
                 classifier_hidden_units: Optional[int] = None,
                 **base_kwargs):
        super().__init__(**base_kwargs)
        self.problem_type = "single_label_classification"
        self.add_pooling_layer = False
        self.return_dict = False
        self.classifier_hidden_units = classifier_hidden_units if classifier_hidden_units else self.hidden_size

BartEncForStanceConfig.register_for_auto_class("AutoConfig")