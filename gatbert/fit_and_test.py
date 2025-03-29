"""
This script primarily exists for development runs
(i.e. we're not using the real test set, just a subset of training or validation).
"""
from .cli import cli_main

if __name__ == "__main__":
    cli = cli_main(run=False)
    cli.datamodule.encoder = cli.model.encoder
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path='best')