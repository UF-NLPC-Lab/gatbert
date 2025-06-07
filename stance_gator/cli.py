# 3rd Party
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
# Local
from .base_module import StanceModule
from .modules import *
from .data_modules import *

class StanceCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """
        I frequently use this, but don't need it for this project yet.
        """
    def after_instantiate_classes(self):
        self.datamodule.encoder = self.model.encoder

def cli_main(**cli_kwargs):
    return StanceCLI(
        model_class=StanceModule, subclass_mode_model=True,
        datamodule_class=StanceDataModule, subclass_mode_data=True,
        trainer_defaults={
            "max_epochs": 1000,
            "deterministic": True
        },
        seed_everything_default=0,
        **cli_kwargs
    )
