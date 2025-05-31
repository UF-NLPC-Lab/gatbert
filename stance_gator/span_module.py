# STL
import copy
from typing import Optional
import dataclasses
# 3rd Party
import torch
from torch.distributions import Categorical
from transformers import BertTokenizerFast
from transformers.utils.generic import ModelOutput
# Local
from .models import BertForStance, BertForStanceConfig
from .encoder import SimpleEncoder, get_text_masks, Encoder
from .constants import DEFAULT_MODEL
from .base_module import StanceModule

class SpanModule(StanceModule):

    @dataclasses.dataclass
    class Output:
        first_pass: BertForStance.Output
        second_pass: Optional[BertForStance.Output] = None
        start_logits: Optional[torch.Tensor] = None
        stop_logits: Optional[torch.Tensor] = None
        start_inds: Optional[torch.Tensor] = None
        stop_inds: Optional[torch.Tensor] = None

        @property
        def logits(self):
            return self.second_pass.logits if self.second_pass else self.first_pass.logits

    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL,
                 classifier_hidden_units: Optional[int] = None,
                 max_context_length: int =256,
                 max_target_length: int =64,
                 warmup_epochs: int = 1,
                 span_weight: float = 1e-3,
                 **parent_kwargs,
                 ):
        super().__init__(**parent_kwargs)
        self.save_hyperparameters()
        self.warmup_epochs = warmup_epochs
        self.span_weight = span_weight
        config = BertForStanceConfig.from_pretrained(pretrained_model,
                                                     classifier_hidden_units=classifier_hidden_units,
                                                     id2label=self.stance_enum.id2label(),
                                                     label2id=self.stance_enum.label2id(),
                                                     )

        self.wrapped = BertForStance.from_pretrained(pretrained_model, config=config)
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(pretrained_model)
        self.__encoder = SpanModule.Encoder(self.tokenizer, max_context_length=max_context_length, max_target_length=max_target_length)

        hidden_size = config.hidden_size
        self.span_ffn = torch.nn.Sequential(
            torch.nn.Dropout(config.hidden_dropout_prob),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 2, bias=True),

            # torch.nn.Linear(hidden_size, 1, bias=True),
            # torch.nn.Flatten(start_dim=-2),
            # torch.nn.Sigmoid()
        )

    def configure_optimizers(self):
        optimizer_bert = torch.optim.Adam(self.wrapped.parameters(), lr=4e-5)
        optimizer_span = torch.optim.Adam(self.span_ffn.parameters(), lr=1e-3)
        return optimizer_bert, optimizer_span

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        # optimizer_bert, optimizer_span = self.optimizers()
        optimizer_bert = self.optimizers()

        two_pass = self.current_epoch >= self.warmup_epochs
        output_obj: SpanModule.Output = self(batch, two_pass=two_pass)

        first_pass = output_obj.first_pass
        ce_first = torch.nn.functional.cross_entropy(first_pass.logits, labels)
        self.log('loss/ce/first', ce_first)

        if output_obj.second_pass is not None:
            second_pass = output_obj.second_pass
            ce_second = torch.nn.functional.cross_entropy(second_pass.logits, labels)
            self.log('loss/ce/second', ce_second)

            loss_diff = ce_second_values - ce_first_values
            seq_lens = output_obj.stop_inds - output_obj.start_inds

            start_log_probs = torch.nn.functional.log_softmax(output_obj.start_logits)[:, output_obj.start_inds]
            stop_log_probs = torch.nn.functional.log_softmax(output_obj.stop_logits)[:, output_obj.stop_inds]
            log_probs = (start_log_probs + stop_log_probs) / 2

            ce_second = second_pass.loss
            self.log('loss/ce/second', ce_second)

            total_ce = ce_first + ce_second
            self.manual_backward(total_ce)
            optimizer_bert.step()

            pass

    def forward(self, batch, two_pass = True):
        context_mask = batch.pop('context_mask')

        first_pass_out = self.wrapped(**batch)
        if not two_pass:
            return SpanModule.Output(first_pass=first_pass_out)

        first_pass_hidden = first_pass_out.last_hidden_state
        bound_logits = self.span_ffn(first_pass_hidden)
        # Only consider context tokens (not target, and not special tokens like [SEP] or [CLS]) as potential endpoints
        bound_logits = bound_logits + torch.where(torch.unsqueeze(context_mask, -1), 0, -torch.inf)


        start_logits = bound_logits[..., 0]
        stop_logits = bound_logits[..., 1]

        # Prepare the spans for the second pass
        with torch.no_grad():
            # TODO: Sample by probability instead of taking the maximum?
            start_inds = torch.argmax(start_logits, dim=-1)
            stop_inds = torch.argmax(stop_logits, dim=-1)

            seq_inds = torch.arange(0, context_mask.shape[1], device=context_mask.device)
            seq_inds = torch.unsqueeze(seq_inds, 0)
            within_range = torch.logical_or(start_inds <= seq_inds, seq_inds <= stop_inds)
            # We && with the context_mask because we only want to add masks for context tokens, not [SEP] or target tokens

            old_att_mask = batch['attention_mask']
            new_att_mask = torch.where(torch.logical_not(context_mask),
                                       old_att_mask,
                                       within_range
            )
            new_batch = copy.copy(batch)
            new_batch['attention_mask'] = new_att_mask

        second_pass_out = self.wrapped(new_batch)
        return SpanModule.Output(first_pass=first_pass_out,
                                 second_pass=second_pass_out,
                                 start_logits=start_logits,
                                 stop_logits=stop_logits,
                                 start_inds=start_inds,
                                 stop_inds=stop_inds)

    @property
    def encoder(self):
        return self.__encoder

    class Encoder(SimpleEncoder):
        def encode(self, sample):
            encoding = super().encode(sample)
            context_mask, _ = get_text_masks(encoding['special_tokens_mask'])
            encoding['context_mask'] = context_mask
            return encoding

    @property
    def feature_size(self) -> int:
        return self.wrapped.config.hidden_size
        

