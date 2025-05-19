# STL
import argparse
# 3rd Party
import lightning as L
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch.loggers import CSVLogger
# Local
from .rgcn import CNEncoder



def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_everything", default=0, type=int)
    parser.add_argument("-cn", metavar="assertions.tsv", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    seed_everything(args.seed_everything)

    # assertions_path = "/home/ethanlmines/blue_dir/datasets/conceptnet/filtered/vast_cn.tsv"
    assertions_path = args.cn
    # out_dir = "graph_logs/"
    mod = CNEncoder(assertions_path)
    logger = CSVLogger(save_dir=args.save_dir, name=None, version=args.version)
    logger.log_hyperparams(mod.hparams)

    trainer = L.Trainer(
        max_epochs=300,
        logger=logger,
        deterministic=True,
        log_every_n_steps=10,
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit(model=mod, train_dataloaders=mod)
    pass

if __name__ == "__main__":
    main()