# STL
import json
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

def grid_iter(**params):
    names = []
    domains = []
    for (name, domain) in params.items():
        names.append(name)
        domains.append(domain)
    for hvals in product(*domains):
        yield dict(zip(names, hvals))

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
    print(f"Chose {val_domain} as the validation set")
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


    best_score = -1
    best_config = {}
    os.makedirs(args.out, exist_ok=True)
    for hparam_dict in grid_iter(adv_weight=[0.25, 0.75], recon_weight=[1e-1, 1], reg_weight=[1e-1, 1, 1e1]):
        print(f"Testing config {hparam_dict}")
        seed_everything(0)

        # FIXME: Really inefficient to re-encode these samples each time
        module = AdvModule(held_out=heldout, **hparam_dict)
        encode = module.encoder.encode
        collate = module.encoder.collate
        train_loader = DataLoader(MapDataset(map(encode, train_samples)), batch_size=DEFAULT_BATCH_SIZE, collate_fn=collate)
        val_loader = DataLoader(MapDataset(map(encode, val_samples)), batch_size=DEFAULT_BATCH_SIZE, collate_fn=collate)

        version = "_".join([f"{name}_{val}" for name,val in hparam_dict.items()])
        logger = CSVLogger(save_dir=args.out,
                           name=f"heldout_{heldout}",
                           version=version)
        trainer = L.Trainer(
                            max_epochs=4,
                            log_every_n_steps=10,
                            enable_checkpointing=False,
                            logger=logger
                            )
        trainer.fit(model=module, train_dataloaders=train_loader)
        [test_scores] = trainer.test(model=module, dataloaders=val_loader)
        with open(os.path.join(logger.log_dir, 'hparams.json'), 'w') as w:
            w.write(json.dumps(hparam_dict))
        macro_f1 = test_scores['test_macro_f1']
        if macro_f1 > best_score:
            print("New best!")
            best_score = macro_f1
            best_config = hparam_dict
    print(f"Best config: {best_config}")
    with open(os.path.join(logger.root_dir, 'best.json'), 'w') as w:
        out_json = {'config': best_config, 'test_macro_f1': best_score}
        w.write(json.dumps(out_json))



if __name__ == "__main__":
    main()
    pass