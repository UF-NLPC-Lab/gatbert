# STL
import csv
from typing import Tuple, Dict, Optional
import pathlib
# 3rd Party
import torch
from torchmetrics.functional.classification import multiclass_stat_scores
from lightning.pytorch.callbacks import Callback
import lightning as L
# Local

class StatsCallback(Callback):

    def __init__(self,
                 label2id: Dict[str, int],
                 results_path: Optional[pathlib.Path] = None):

        assert label2id
        assert sorted(label2id.values()) == list(range(len(label2id)))
        self.__label2id = label2id

        self.__stats = torch.zeros((len(label2id), 5), dtype=torch.int)
        self.__results = dict()
        self.__summarized = False

        self.__results_path = results_path
        self.__res_file = None
        self.__writer = None

    @staticmethod
    def compute_metrics(tp, fp, fn) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute precision, recall, and f1
        """
        pred_pos = tp + fp
        precision = tp / pred_pos if pred_pos > 0 else 0
        support = tp + fn
        recall = tp / support if support > 0 else 0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom > 0 else 0
        return precision, recall, f1


    def record(self, preds: torch.Tensor, labels: torch.Tensor):
        if self.__summarized:
            raise ValueError("Must reset F1Calc before recording more results")
        batch_stats = \
            multiclass_stat_scores(preds, labels, num_classes=len(self.__label2id), average='none')
        self.__stats += batch_stats.to(self.__stats.device)

    def reset(self):
        self.__stats = torch.zeros((len(self.__label2id), 5), dtype=torch.int)
        self.__results = dict()
        self.__summarized = False

    @property
    def results(self):
        return self.__results


    def on_test_epoch_start(self, trainer, pl_module):
        self.reset()
        if self.__results_path is not None:
            self.__res_file = open(self.__results_path, 'w')
            self.__writer = csv.writer(self.__res_file)
            self.__writer.writerow(["pred", "label"])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        labels = batch['labels']
        # FIXME: support preds, not just logits
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        self.record(preds, labels)

        if self.__writer is not None:
            preds = preds.detach().cpu().tolist()
            labels = labels.detach().cpu().tolist()
            self.__writer.writerows(zip(preds, labels))

    def on_test_epoch_end(self, trainer, pl_module):
        rval = self._on_epoch_end(trainer, pl_module, "test")
        if self.__res_file is not None:
            self.__res_file.close()
        return rval

    def _on_epoch_end(self, trainer, pl_module: L.LightningModule, stage):
        results = {}
        for (label, ind) in self.__label2id.items():
            results[f'class_{label}_precision'], results[f'class_{label}_recall'], results[f'class_{label}_f1'] = \
                StatsCallback.compute_metrics(self.__stats[ind, 0], self.__stats[ind, 1], self.__stats[ind, 3])
        agg_counts = torch.sum(self.__stats, dim=0)
        _, _2, results['micro_f1'] = StatsCallback.compute_metrics(agg_counts[0], agg_counts[1], agg_counts[3])
        results['macro_f1'] = sum(results[f'class_{label}_f1'] for label in self.__label2id) / len(self.__label2id)
        results = {f"{stage}_{k}":v for k,v in results.items()}
        self.__results = results

        if stage != 'predict':
            for (k, v) in self.__results.items():
                pl_module.log(k, v, on_step=False, on_epoch=True)
