# 3rd Party
import torch
import lightning as L
from transformers import AutoConfig, AutoModel
# Local
from .f1_calc import F1Calc
from .constants import DEFAULT_MODEL, NUM_CN_RELATIONS, DEFAULT_ATT_TYPE
from .stance_classifier import GraphClassifier, BertClassifier, ConcatClassifier, GraphOnlyClassifier
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

class GraphOnlyStanceModule(StanceModule):
    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL,
                 att_type: AttentionType = DEFAULT_ATT_TYPE):
        pass
        super().__init__()
        self.save_hyperparameters()
        model_config = GatbertConfig(
            AutoConfig.from_pretrained(self.hparams.pretrained_model),
            n_relations=NUM_CN_RELATIONS,
            att_type=self.hparams.att_type)
        model_config.num_hidden_layers = 2 # TODO: Make configurable
        self.__classifier = GraphOnlyClassifier(pretrained_model, model_config)
    def forward(self, *args, **kwargs):
        return self.__classifier(*args, **kwargs)

class GraphStanceModule(StanceModule):
    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL,
                 load_pretrained_weights: bool = True,
                 att_type: AttentionType = DEFAULT_ATT_TYPE):
        super().__init__()
        self.save_hyperparameters()

        model_config = GatbertConfig(AutoConfig.from_pretrained(self.hparams.pretrained_model), n_relations=NUM_CN_RELATIONS, att_type=self.hparams.att_type)
        self.__classifier = GraphClassifier(model_config)
        if self.hparams.load_pretrained_weights:
            orig_model = AutoModel.from_pretrained(self.hparams.pretrained_model)
            self.__classifier.bert.load_pretrained_weights(orig_model)

    def forward(self, *args, **kwargs):
        return self.__classifier(*args, **kwargs)

class ConcatStanceModule(StanceModule):
    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL,
                 att_type: AttentionType = 'edge_as_att'):
        super().__init__()
        self.save_hyperparameters()
        model_config = GatbertConfig(AutoConfig.from_pretrained(self.hparams.pretrained_model),
                                    n_relations=NUM_CN_RELATIONS,
                                    att_type=self.hparams.att_type)
        model_config.num_hidden_layers = 2 # TODO: Make configurable once it matters
        self.__classifier = ConcatClassifier(pretrained_model, model_config)
    def forward(self, *args, **kwargs):
        return self.__classifier(*args, **kwargs)
    
class BertStanceModule(StanceModule):
    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL):
        super().__init__()
        self.save_hyperparameters()
        self.__classifier = BertClassifier(pretrained_model)

    def forward(self, *args, **kwargs):
        return self.__classifier(*args, **kwargs)