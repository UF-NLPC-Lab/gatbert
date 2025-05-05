import abc
# 3rd Party
import torch
import lightning as L
import typing
# Local
from .f1_calc import F1Calc
from .encoder import Encoder
from .output import StanceOutput

class StanceModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.__calc = F1Calc()

    @property
    @abc.abstractmethod
    def encoder(self) -> Encoder:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def feature_size(self) -> int:
        raise NotImplementedError

    def get_optimizer_params(self):
        return [{"params": self.parameters(), "lr": 4e-5}]

    def configure_optimizers(self):
        return torch.optim.Adam(self.get_optimizer_params())
    def training_step(self, batch, batch_idx):
        # Calls the forward method defined in subclass
        result = self(**batch)
        loss = typing.cast(StanceOutput, result).loss
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
        labels = batch.pop('labels').view(-1)
        rval = self(**batch)
        logits = typing.cast(StanceOutput, rval).logits
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