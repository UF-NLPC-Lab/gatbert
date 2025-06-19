import typing
import pathlib
# 3rd Party
import torch
from lightning.pytorch.cli import LightningArgumentParser
# Local
from .cli import StanceCLI
from .modules import StanceModule
from .data import VizDataModule

class VizCLI(StanceCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_argument("--ckpt", type=pathlib.Path, metavar="model.ckpt", required=True, help="Model checkpoint")
        parser.add_argument("-o", type=pathlib.Path, metavar="output/", required=True, help="Output directory")

def main(raw_args=None):
    cli = VizCLI(
        model_class=StanceModule, subclass_mode_model=True,
        datamodule_class=VizDataModule, subclass_mode_data=True,
        seed_everything_default=0,
        run=False,
        trainer_defaults={
            "logger": False
        }
    )

    model = typing.cast(StanceModule, cli.model)
    model.load_state_dict(torch.load(cli.config.ckpt)['state_dict'], strict=False)
    datamodule = typing.cast(VizDataModule, cli.datamodule)

    out_dir = typing.cast(pathlib.Path, cli.config.o)
    out_dir.mkdir(exist_ok=True)
    cli.trainer.callbacks.append(model.make_visualizer(out_dir))
    predictions = cli.trainer.predict(model, datamodule)



if __name__ == "__main__":
    main()
    pass