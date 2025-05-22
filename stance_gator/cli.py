# 3rd Party
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
# Local
from .base_module import StanceModule
from .modules import *
from .data_modules import *

class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """
        I frequently use this, but don't need it for this project yet.
        """

def cli_main(**cli_kwargs):
    return CustomCLI(
        model_class=StanceModule, subclass_mode_model=True,
        datamodule_class=StanceDataModule, subclass_mode_data=True,
        trainer_defaults={
            "max_epochs": 1000,
            "deterministic": True
        },
        **cli_kwargs
    )
