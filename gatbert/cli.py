# 3rd Party
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
# Local
from .modules import *
from .data_modules import *
from .stance_classifier import *

class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """
        I frequently use this, but don't need it for this project yet.
        """

def cli_main(**cli_kwargs):

    STOPPING_METRIC = "val_macro_f1"
    model_callback = ModelCheckpoint(
        monitor=STOPPING_METRIC,
        mode='max',
        filename="{epoch:02d}-{val_macro_f1:.3f}"
    )
    early_stopping_callback = EarlyStopping(
        monitor=STOPPING_METRIC,
        patience=3,
        mode='max'
    )

    return CustomCLI(
        model_class=MyStanceModule, subclass_mode_model=False,
        datamodule_class=StanceDataModule, subclass_mode_data=True,
        trainer_defaults={
            "max_epochs": 1000,
            "callbacks": [
                model_callback,
                early_stopping_callback
            ]
        },
        **cli_kwargs
    )
