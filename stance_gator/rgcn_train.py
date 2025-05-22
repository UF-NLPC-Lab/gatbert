# STL
# 3rd Party
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
# Local
from .rgcn import CNEncoder

def main(raw_args=None):
    cli = LightningCLI(
        model_class=CNEncoder,
        seed_everything_default=0,
        run=False,
        trainer_defaults={
            "callbacks": [
                ModelCheckpoint(filename="{epoch:02d}-{loss:.3f}", mode='min', monitor='loss'),
                EarlyStopping(monitor="loss", mode='min', patience=10)
            ],
            "gradient_clip_algorithm": 'norm',
            "gradient_clip_val": 1.0,
            "reload_dataloaders_every_n_epochs": 1
        }
    )
    cli.trainer.fit(model=cli.model)

if __name__ == "__main__":
    main()