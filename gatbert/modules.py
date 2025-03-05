from typing import Optional, Dict
import inspect
# 3rd Party
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import lightning as L
from transformers import AutoConfig
# Local
from .f1_calc import F1Calc
from .constants import DEFAULT_MODEL, NUM_CN_RELATIONS, DEFAULT_ATT_TYPE, DEFAULT_BATCH_SIZE
from .stance_classifier import TextClassifier, StanceClassifier
from .config import GatbertConfig
from .types import AttentionType, CorpusType
from .data import parse_ez_stance, parse_graph_tsv, parse_semeval, parse_vast, MapDataset
from .utils import map_func_gen

class StanceModule(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.__ce = torch.nn.CrossEntropyLoss()
        self.__calc = F1Calc()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)
    def training_step(self, batch, batch_idx):
        labels = batch.pop("stance")
        # Calls the forward method defined in subclass
        logits = self(**batch)
        loss = self.__ce(logits, labels)
        self.log("loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        self.__eval_step(batch, batch_idx)
    def test_step(self, batch, batch_idx):
        self.__eval_step(batch, batch_idx)
    def on_validation_epoch_end(self):
        self.__eval_finish('val')
    def on_test_epoch_end(self):
        self.__eval_finish('test')


    def __eval_step(self, batch, batch_idx):
        labels = batch.pop('stance').view(-1)
        logits = self(**batch)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        self.__calc.record(probs, labels)
    def __eval_finish(self, stage):
        self.__log_stats(self.__calc, f"{stage}")
    def __log_stats(self, calc: F1Calc, prefix):
        calc.summarize()
        self.log(f"{prefix}_favor_precision", calc.favor_precision)
        self.log(f"{prefix}_favor_recall", calc.favor_recall)
        self.log(f"{prefix}_favor_f1", calc.favor_f1)
        self.log(f"{prefix}_against_precision", calc.against_precision)
        self.log(f"{prefix}_against_recall", calc.against_recall)
        self.log(f"{prefix}_against_f1", calc.against_f1)
        self.log(f"{prefix}_macro_f1", calc.macro_f1)
        calc.reset()


class MyStanceModule(StanceModule):
    def __init__(self,
                 classifier: StanceClassifier,
                 pretrained_model: str = DEFAULT_MODEL,
                 att_type: AttentionType = DEFAULT_ATT_TYPE,
                 num_graph_layers: Optional[int] = None,
                 load_pretrained_weights: bool = True,
                 graph: Optional[str] = None,

                 # Data related
                 corpus_type: CorpusType = 'graph',
                 batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        super().__init__()
        self.save_hyperparameters()
        model_config = GatbertConfig(
            AutoConfig.from_pretrained(self.hparams.pretrained_model),
            n_relations=NUM_CN_RELATIONS,
            num_graph_layers=self.hparams.num_graph_layers,
            att_type=self.hparams.att_type,
            base_model=self.hparams.pretrained_model,
        )
        self.classifier = classifier
        # Can't do self.hparams.classifier like I normally would becaues that's still a string...?
        if self.hparams.load_pretrained_weights:
            self.classifier.load_pretrained_weights()

        self.__data: Dict[str, MapDataset] = {}
        self.__train_ds: Dataset = None
        self.__val_ds: Dataset = None
        self.__test_ds: Dataset = None

    def forward(self, *args, **kwargs):
        return self.__classifier(*args, **kwargs)

    # Data members
    def prepare_data(self):
        corpus_type = self.hparams.corpus_type
        if corpus_type == 'graph':
            parse_fn = parse_graph_tsv
        elif corpus_type == 'ezstance':
            parse_fn = parse_ez_stance
        elif corpus_type == 'semeval':
            parse_fn = parse_semeval
        elif corpus_type == 'vast':
            parse_fn = parse_vast
        else:
            raise ValueError(f"Invalid corpus_type {corpus_type}")

        transforms = self.hparams.transforms
        if transforms:
            transform_map = {
                'rm_external': lambda x: x.strip_external()
            }
            for t in transforms:
                if t in transform_map:
                    parse_fn = map_func_gen(transform_map[t], parse_fn)
        parse_fn = map_func_gen(self.classifier.encode, parse_fn)
        for data_path in self.hparams.partitions:
            self.__data[data_path] = list(parse_fn, data_path)
    def setup(self, stage):
        train_dses = []
        val_dses = []
        test_dses = []
        for (data_prefix, (train_frac, val_frac, test_frac)) in self.hparams.partitions.items():
            train_ds, val_ds, test_ds = \
                random_split(self.__data[data_prefix], [train_frac, val_frac, test_frac])
            train_dses.append(train_ds)
            val_dses.append(val_ds)
            test_dses.append(test_ds)
        self.__train_ds = ConcatDataset(train_dses)
        self.__val_ds = ConcatDataset(val_dses)
        self.__test_ds = ConcatDataset(test_dses)
    def train_dataloader(self):
        return DataLoader(self.__train_ds, batch_size=self.hparams.batch_size, collate_fn=self._preprocessor.collate)
    def val_dataloader(self):
        return DataLoader(self.__val_ds, batch_size=self.hparams.batch_size, collate_fn=self._preprocessor.collate)
    def test_dataloader(self):
        return DataLoader(self.__test_ds, batch_size=self.hparams.batch_size, collate_fn=self._preprocessor.collate)