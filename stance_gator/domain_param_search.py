# STL
import os
import argparse
from itertools import product, chain
# 3rd party
import lightning as L
import numpy as np
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
# Local
from .data import parse_ez_stance, MapDataset
from .constants import EzstanceDomains, DEFAULT_BATCH_SIZE
from .adv_module import AdvModule

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, metavar="ezstance/subtaskB/(claim|mixed|noun_phrase)")
    parser.add_argument("--out", required=True, metavar="results_dir/")
    parser.add_argument("--heldout", required=True, metavar="politcs|covid19_domain|...")

    args = parser.parse_args(raw_args)

    # Make my own training and validation split (but use same test split) based on a held-out target
    heldout = args.heldout
    domains = [dom.value for dom in EzstanceDomains]
    assert heldout in domains
    rem_domains = [dom for dom in domains if dom != heldout]
    np_rng = np.random.default_rng(seed=0)
    np_rng.shuffle(rem_domains)
    val_domain = np_rng.choice(rem_domains)
    data_dir = os.path.join(args.data, args.heldout)
    train_iter = parse_ez_stance(os.path.join(data_dir, "raw_train_all_onecol.csv"))
    val_iter = parse_ez_stance(os.path.join(data_dir, "raw_val_all_onecol.csv"))
    train_samples = []
    val_samples = []
    for s in chain(train_iter, val_iter):
        if s.domain == val_domain:
            val_samples.append(s)
        else:
            train_samples.append(s)


    os.makedirs(args.out, exist_ok=True)
    adv_weights   = [0.25, 0.75]
    recon_weights = [1e-1, 1]
    reg_weights   = [1e-1, 1, 1e1]
    for (adv_weight, recon_weight, reg_weight) in product(adv_weights, recon_weights, reg_weights):
        seed_everything(0)

        # FIXME: Really inefficient to re-encode these samples each time
        module = AdvModule(held_out=heldout,
                           adv_weight=adv_weight,
                           recon_weight=recon_weight,
                           reg_weight=reg_weight)
        encode = module.encoder.encode
        collate = module.encoder.collate
        train_loader = DataLoader(MapDataset(map(encode, train_samples)), batch_size=DEFAULT_BATCH_SIZE, collate_fn=collate)
        val_loader = DataLoader(MapDataset(map(encode, val_samples)), batch_size=DEFAULT_BATCH_SIZE, collate_fn=collate)

        logger = CSVLogger(save_dir=args.out,
                           name=f"heldout_{heldout}",
                           version=f"adv_{adv_weight}_rec_{recon_weight}_reg_{reg_weight}",)
        trainer = L.Trainer(max_epochs=1, log_every_n_steps=10, enable_checkpointing=False, logger=logger)
        trainer.fit(model=module, train_dataloaders=train_loader)
        trainer.test(model=module, dataloaders=val_loader)

        break



if __name__ == "__main__":
    main()
    pass