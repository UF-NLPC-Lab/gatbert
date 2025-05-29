import abc
# 3rd Party
import torch
import lightning as L
import typing
# Local
from .f1_calc import F1Calc
from .encoder import Encoder
from .output import StanceOutput
from .constants import StanceType, STANCE_TYPE_MAP

class StanceModule(L.LightningModule):
    def __init__(self, stance_type: StanceType = 'tri'):
        super().__init__()
        self.stance_enum = STANCE_TYPE_MAP[stance_type]
        self._calc = F1Calc(self.stance_enum.label2id())

    @property
    @abc.abstractmethod
    def encoder(self) -> Encoder:
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
        self._eval_step(batch, batch_idx)
    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx)
    def on_validation_epoch_end(self):
        self.__eval_finish('val')
    def on_test_epoch_end(self):
        self.__eval_finish('test')


    def _eval_step(self, batch, batch_idx):
        labels = batch.pop('labels').view(-1)
        rval = self(**batch)
        logits = typing.cast(StanceOutput, rval).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        self._calc.record(probs, labels)

    def __eval_finish(self, stage):
        self.__log_stats(self._calc, f"{stage}")
    def __log_stats(self, calc: F1Calc, prefix):
        calc.summarize()
        for class_name in self.stance_enum.label2id():
            k = f'{class_name}_f1'
            self.log(f'{prefix}_{k}', calc.results[k])
        self.log(f'{prefix}_macro_f1', calc.results['macro_f1'])
        calc.reset()

    # FIXME: Figure out the more standard way to do this
    def on_train_epoch_start(self):
        self.train()
    def on_validation_epoch_start(self):
        self.eval()
    def on_test_epoch_start(self):
        self.eval()