# STL
from typing import Optional, Tuple
# 3rd Party
import torch
from torchmetrics.functional.classification import multiclass_stat_scores
# Local
from .constants import Stance

class F1Calc:
    def __init__(self):
        self.__favor_stats = torch.zeros(5, dtype=torch.int)
        self.__against_stats = torch.zeros(5, dtype=torch.int)
        self.__neutral_stats = torch.zeros(5, dtype=torch.int)
        self.__summarized = False

        self.favor_precision: Optional[float] = None
        self.favor_recall: Optional[float] = None
        self.favor_f1: Optional[float] = None
        self.against_precision: Optional[float] = None
        self.against_recall: Optional[float] = None
        self.against_f1: Optional[float] = None
        self.neutral_precision: Optional[float] = None
        self.neutral_recall: Optional[float] = None
        self.neutral_f1: Optional[float] = None
        self.macro_f1: Optional[float] = None

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
        stats = self.__stat_func(preds, labels).to(self.__favor_stats.device)
        self.__favor_stats += stats[Stance.FAVOR.value]
        self.__against_stats += stats[Stance.AGAINST.value]
        self.__neutral_stats += stats[Stance.NONE.value]

    def reset(self):
        self.__favor_stats = torch.zeros(5, dtype=torch.int)
        self.__against_stats = torch.zeros(5, dtype=torch.int)
        self.__neutral_stats = torch.zeros(5, dtype=torch.int)
        self.__summarized = False

    def summarize(self):
        self.favor_precision, self.favor_recall, self.favor_f1 = \
            F1Calc.compute_metrics(self.__favor_stats[0], self.__favor_stats[1], self.__favor_stats[3])
        self.against_precision, self.against_recall, self.against_f1 = \
            F1Calc.compute_metrics(self.__against_stats[0], self.__against_stats[1], self.__against_stats[3])
        self.neutral_precision, self.neutral_recall, self.neutral_f1 = \
            F1Calc.compute_metrics(self.__neutral_stats[0], self.__neutral_stats[1], self.__neutral_stats[3])
        self.macro_f1 = (self.favor_f1 + self.against_f1 + self.neutral_f1) / 3
        self.__summarized = True

    def __stat_func(self, preds, targets):
        return multiclass_stat_scores(preds, targets, num_classes=len(Stance), average='none')
