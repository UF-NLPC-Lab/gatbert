# STL
from typing import Tuple, Dict
# 3rd Party
import torch
from torchmetrics.functional.classification import multiclass_stat_scores
# Local

class F1Calc:
    def __init__(self, label2id: Dict[str, int]):

        assert label2id
        assert sorted(label2id.values()) == list(range(len(label2id)))
        self.__label2id = label2id

        self.__stats = torch.zeros((len(label2id), 5), dtype=torch.int)
        self.__results = dict()
        self.__summarized = False

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
        self.__stats += self.__stat_func(preds, labels).to(self.__stats.device)

    def reset(self):
        self.__stats = torch.zeros((len(self.__label2id), 5), dtype=torch.int)
        self.__results = dict()
        self.__summarized = False

    @property
    def results(self):
        return self.__results

    def summarize(self):
        results = self.__results
        for (label, ind) in self.__label2id.items():
            results[f'{label}_precision'], results[f'{label}_recall'], results[f'{label}_f1'] = \
                F1Calc.compute_metrics(self.__stats[ind, 0], self.__stats[ind, 1], self.__stats[ind, 3])
        agg_counts = torch.sum(self.__stats, dim=0)
        _, _2, results['micro_f1'] = F1Calc.compute_metrics(agg_counts[0], agg_counts[1], agg_counts[3])
        results['macro_f1'] = sum(results[f'{label}_f1'] for label in self.__label2id) / len(self.__label2id)
        self.__summarized = True

    def __stat_func(self, preds, targets):
        return multiclass_stat_scores(preds, targets, num_classes=len(self.__label2id), average='none')
