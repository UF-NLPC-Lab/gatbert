# 3rd Party
import torch
import lightning as L
# Local
from .f1_calc import F1Calc
from .stance_classifier import *

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
        self.log(f"{prefix}_favor_f1", calc.favor_f1)
        self.log(f"{prefix}_against_f1", calc.against_f1)
        self.log(f"{prefix}_neutral_f1", calc.neutral_f1)
        self.log(f"{prefix}_macro_f1", calc.macro_f1)
        calc.reset()

    # FIXME: Figure out the more standard way to do this
    def on_train_epoch_start(self):
        self.train()
    def on_validation_epoch_start(self):
        self.eval()
    def on_test_epoch_start(self):
        self.eval()

class MyStanceModule(StanceModule):
    """
    Simple wrapper around my non-Lightning model classes.
    
    I want maximum configurability of the model classes,
    but I also don't want them to depend on Lightning itself.
    This wrapper allows for that.
    """
    def __init__(self,
                 classifier: StanceClassifier,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.classifier = classifier
        # FIXME: Do this in a cleaner way
        self.train()
    def forward(self, *args, **kwargs):
        return self.classifier(*args, **kwargs)

