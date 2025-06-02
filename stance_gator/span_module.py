# STL
import copy
from typing import Optional
import dataclasses
import typing
# 3rd Party
import torch
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
                 max_context_length: int = 256,
                 max_target_length: int = 64,
                 warmup_epochs: int = 0,
                 ce_weight: float = 1e-3,
                 span_weight: float = 1e-3,
                 **parent_kwargs,
                 ):
        super().__init__(**parent_kwargs)
        self.save_hyperparameters()
        self.warmup_epochs = warmup_epochs
        self.ce_weight = ce_weight
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

        )

    @property
    def _greedy(self):
        return not self.training

    def _eval_step(self, batch, batch_idx, stage):
        orig_lens = torch.sum(batch['context_mask'], dim=-1)

        labels = batch.pop('labels').view(-1)
        rval: SpanModule.Output = self(**batch)
        logits = rval.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        self._calc.record(probs, labels)

        seq_lens = torch.nn.functional.relu(rval.stop_inds - rval.start_inds + 1).to(logits.dtype)
        self.log(f"{stage}_seqlen", torch.mean(seq_lens), batch_size=seq_lens.numel(), on_epoch=True)

        seq_reduction = (orig_lens - seq_lens) / orig_lens
        self.log(f"{stage}_seqreduction", torch.mean(seq_reduction), batch_size=seq_reduction.numel(), on_epoch=True)



    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        output_obj: SpanModule.Output = self(**batch)

        first_pass = output_obj.first_pass
        ce_first = torch.nn.functional.cross_entropy(first_pass.logits, labels)
        self.log('train_ce_first', ce_first)

        if output_obj.second_pass is not None:
            second_pass = output_obj.second_pass
            ce_second_vals = torch.nn.functional.cross_entropy(second_pass.logits, labels, reduction='none')
            self.log('train_ce_second', torch.mean(ce_second_vals))

            seq_lens = torch.nn.functional.relu(output_obj.stop_inds - output_obj.start_inds + 1).to(ce_second_vals.dtype)
            self.log('train_seqlen', torch.mean(seq_lens))

            start_log_probs = torch.nn.functional.log_softmax(output_obj.start_logits, dim=-1)
            stop_log_probs = torch.nn.functional.log_softmax(output_obj.stop_logits, dim=-1)
            start_log_probs = torch.gather(start_log_probs, dim=-1, index=torch.unsqueeze(output_obj.start_inds, -1))
            stop_log_probs = torch.gather(stop_log_probs, dim=-1, index=torch.unsqueeze(output_obj.stop_inds, -1))
            start_log_probs = torch.squeeze(start_log_probs, -1)
            stop_log_probs = torch.squeeze(stop_log_probs, -1)

            log_probs = (start_log_probs + stop_log_probs) / 2
            self.log('train_span_logprob', torch.mean(log_probs))

            rl_loss = -log_probs * (self.ce_weight * ce_second_vals + self.span_weight * seq_lens)
            rl_loss = torch.mean(rl_loss)
            self.log("train_rl_loss", rl_loss)
            loss = ce_first + rl_loss
        else:
            loss = ce_first

        self.log('train_loss', loss)
        return loss

    def forward(self, **batch):
        two_pass = batch.pop('two_pass', self.current_epoch >= self.warmup_epochs)
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

            if self._greedy:
                start_inds = torch.argmax(start_logits, dim=-1)
                stop_inds = torch.argmax(stop_logits, dim=-1)
            else:
                start_dist = torch.distributions.Categorical(logits=start_logits)
                stop_dist = torch.distributions.Categorical(logits=stop_logits)
                start_inds = start_dist.sample()
                stop_inds = stop_dist.sample()


            seq_inds = torch.arange(0, context_mask.shape[1], device=context_mask.device)
            seq_inds = torch.unsqueeze(seq_inds, 0)
            within_range = torch.logical_and(torch.unsqueeze(start_inds, -1) <= seq_inds,
                                            seq_inds <= torch.unsqueeze(stop_inds, -1) )
            # We && with the context_mask because we only want to add masks for context tokens, not [SEP] or target tokens

            old_att_mask = batch['attention_mask']
            new_att_mask = torch.where(torch.logical_not(context_mask),
                                       old_att_mask,
                                       within_range
            )
            new_batch = copy.copy(batch)
            new_batch['attention_mask'] = new_att_mask

        second_pass_out = self.wrapped(**new_batch)
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
        

