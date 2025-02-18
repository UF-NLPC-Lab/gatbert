from typing import Optional
# 3rd Party
import torch
import lightning as L
from transformers import AutoConfig
# Local
from .f1_calc import F1Calc
from .constants import DEFAULT_MODEL, NUM_CN_RELATIONS, DEFAULT_ATT_TYPE
from .stance_classifier import TextClassifier, StanceClassifier
from .config import GatbertConfig
from .types import AttentionType

class StanceModule(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.__ce = torch.nn.CrossEntropyLoss()
        self.__calc = F1Calc()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)
    def training_step(self, batch, batch_idx):
        labels = batch.pop("stance")
        # Calls the forward method defined in subclass
        logits = self(**batch)
        loss = self.__ce(logits, labels)
        self.log("loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        self.__eval_step(batch, batch_idx)
    def test_step(self, batch, batch_idx):
        self.__eval_step(batch, batch_idx)
    def on_validation_epoch_end(self):
        self.__eval_finish('val')
    def on_test_epoch_end(self):
        self.__eval_finish('test')


    def __eval_step(self, batch, batch_idx):
        labels = batch.pop('stance').view(-1)
        logits = self(**batch)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        self.__calc.record(probs, labels)
    def __eval_finish(self, stage):
        self.__log_stats(self.__calc, f"{stage}")
    def __log_stats(self, calc: F1Calc, prefix):
        calc.summarize()
        self.log(f"{prefix}_favor_precision", calc.favor_precision)
        self.log(f"{prefix}_favor_recall", calc.favor_recall)
        self.log(f"{prefix}_favor_f1", calc.favor_f1)
        self.log(f"{prefix}_against_precision", calc.against_precision)
        self.log(f"{prefix}_against_recall", calc.against_recall)
        self.log(f"{prefix}_against_f1", calc.against_f1)
        self.log(f"{prefix}_macro_f1", calc.macro_f1)
        calc.reset()


class MyStanceModule(StanceModule):
    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL,
                 att_type: AttentionType = DEFAULT_ATT_TYPE,
                 num_graph_layers: Optional[int] = None,
                 classifier: type[StanceClassifier] = TextClassifier,
                 load_pretrained_weights: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        model_config = GatbertConfig(
            AutoConfig.from_pretrained(self.hparams.pretrained_model),
            n_relations=NUM_CN_RELATIONS,
            num_graph_layers=self.hparams.num_graph_layers,
            att_type=self.hparams.att_type,
            base_model=self.hparams.pretrained_model,
        )
        # Can't do self.hparams.classifier like I normally would becaues that's still a string...?
        self.__classifier: StanceClassifier = classifier(model_config)
        if self.hparams.load_pretrained_weights:
            self.__classifier.load_pretrained_weights()
    def forward(self, *args, **kwargs):
        return self.__classifier(*args, **kwargs)