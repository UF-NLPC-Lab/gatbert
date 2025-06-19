import abc
# 3rd Party
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
import typing
import os
# Local
from ..data import Encoder, StanceType, STANCE_TYPE_MAP
from ..output import StanceOutput

class StanceModule(L.LightningModule):
    def __init__(self, stance_type: StanceType = 'tri'):
        super().__init__()
        self.stance_enum = STANCE_TYPE_MAP[stance_type]

    @property
    @abc.abstractmethod
    def encoder(self) -> Encoder:
        raise NotImplementedError

    def make_visualizer(self, output_dir: os.PathLike) -> Callback:
        return Callback()

    def get_optimizer_params(self):
        return [{"params": self.parameters(), "lr": 4e-5}]

    def configure_optimizers(self):
        return torch.optim.Adam(self.get_optimizer_params())
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(**batch)
    def training_step(self, batch, batch_idx):
        # Calls the forward method defined in subclass
        result = self(**batch)
        loss = typing.cast(StanceOutput, result).loss
        self.log("loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'val')
    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'test')
    def _eval_step(self, batch, batch_idx, stage):
        return self(**batch)

    # FIXME: Figure out the more standard way to do this
    def on_train_epoch_start(self):
        self.train()
    def on_validation_epoch_start(self):
        self.eval()
    def on_test_epoch_start(self):
        self.eval()