# STL
from __future__ import annotations
from typing import Optional, Literal, List
import dataclasses
import pathlib
import html
# 3rd Party
from lightning.pytorch.callbacks import Callback
import torch_scatter
import torch
from transformers import BertTokenizerFast, PreTrainedTokenizerFast
from IPython.display import display, HTML
# Local
from .types import TensorDict
from .data import SPACY_PIPES
from .sample import Sample
from .models import BertForStance, BertForStanceConfig
from .encoder import SimpleEncoder, Encoder, keyed_scalar_stack
from .constants import DEFAULT_MODEL
from .base_module import StanceModule

class AggModule(StanceModule):

    @dataclasses.dataclass
    class Output:
        full_logits: torch.Tensor
        agg_logits: torch.Tensor
        sub_logits: torch.Tensor

        @property
        def logits(self):
            return self.agg_logits

    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL,
                 classifier_hidden_units: Optional[int] = None,
                 agg: Literal['mean'] = 'mean',
                 **parent_kwargs,
                 ):
        super().__init__(**parent_kwargs)
        self.save_hyperparameters()
        self.agg = agg
        config = BertForStanceConfig.from_pretrained(pretrained_model,
                                                     classifier_hidden_units=classifier_hidden_units,
                                                     id2label=self.stance_enum.id2label(),
                                                     label2id=self.stance_enum.label2id(),
                                                     )

        self.wrapped = BertForStance.from_pretrained(pretrained_model, config=config)
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(pretrained_model)
        self.__encoder = AggModule.Encoder(self.tokenizer)

        hidden_size = config.hidden_size
        self.span_ffn = torch.nn.Sequential(
            torch.nn.Dropout(config.hidden_dropout_prob),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 2, bias=True),

        )


    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        output_obj: AggModule.Output = self(**batch)


        ce_full = torch.nn.functional.cross_entropy(output_obj.full_logits, labels)
        ce_agg  = torch.nn.functional.cross_entropy(output_obj.agg_logits, labels)
        self.log("train_ce_full", ce_full)
        self.log("train_ce_agg", ce_agg)

        # Note this means for backprop purposes, 1 full sample carries the same weight as all its children
        loss = (ce_full + ce_agg) / 2

        # Alternate weighting scheme
        # Here, 50-something child samples carry more weight in the average than 8 parent samples
        # parent_samples = labels.numel()
        # child_samples = batch['parent'].numel()
        # loss = (parent_samples * ce_full + child_samples * ce_agg) / (parent_samples + child_samples)

        self.log("train_ce", loss)
        return loss

    def forward(self, full, sub, parent, labels=None):

        full_pass: BertForStance.Output = self.wrapped(**full)
        sub_pass: BertForStance.Output = self.wrapped(**sub)

        assert self.agg == 'mean'
        sub_probs = torch.nn.functional.softmax(sub_pass.logits, dim=-1)
        agg_probs = torch_scatter.scatter(sub_probs, parent, dim=-2, reduce='mean')
        agg_log = torch.log(agg_probs) # TODO: Do something more numerically stable
        return AggModule.Output(full_logits=full_pass.logits,
                                agg_logits=agg_log,
                                sub_logits=sub_pass.logits)
    @property
    def encoder(self) -> AggModule.Encoder:
        return self.__encoder

    def make_visualizer(self, output_dir):
        return AggModule.Visualizer(output_dir, self.tokenizer)

    class Encoder(Encoder):
        def __init__(self, tokenizer: PreTrainedTokenizerFast):
            self.tokenizer = tokenizer
            self.nested = SimpleEncoder(tokenizer, max_context_length=None, max_target_length=None)

        def encode_with_meta(self, sample: Sample):
            full_context = " ".join(sample.context) if sample.is_split_into_words else sample.context
            target = " ".join(sample.target) if sample.is_split_into_words else sample.target
            sample = Sample(context=full_context,
                            target=target,
                            is_split_into_words=False,
                            lang=sample.lang,
                            stance=sample.stance)

            lang = sample.lang or 'en'
            pipeline = SPACY_PIPES[lang]

            span_indices = []
            spacy_doc = pipeline(full_context)

            subsample_encodings = []
            for sent in spacy_doc.sents:
                sub_sample = Sample(context=str(sent),
                                    target=target,
                                    stance=sample.stance,
                                    is_split_into_words=False,
                                    lang=lang)
                subsample_encoding = self.nested.encode(sub_sample)
                subsample_encoding.pop('labels')
                subsample_encodings.append(subsample_encoding)
                span_indices.append( (sent.start, sent.end) )

            full_encoding = self.nested.encode(sample)
            label = full_encoding.pop('labels')
            encode_dict = {
                "full": full_encoding,
                "sub": subsample_encodings,
                "labels": label
            }
            meta = span_indices
            return encode_dict, meta

        def collate(self, samples: List[TensorDict]):
            parent_inds = []
            for i, sample in enumerate(samples):
                num_subsamples = len(sample['sub'])
                parent_inds.extend(i for _ in range(num_subsamples))
            parent_inds = torch.tensor(parent_inds)

            full_batch = self.nested.collate([s['full'] for s in samples])
            subsample_batch = self.nested.collate([sub_samp for s in samples for sub_samp in s['sub']])
            labels = keyed_scalar_stack(samples, 'labels')

            return {
                "full": full_batch,
                "sub": subsample_batch,
                "labels": labels,
                "parent": parent_inds
            }


        def encode(self, sample: Sample):
            return self.encode_with_meta(sample)[0]

    @property
    def feature_size(self) -> int:
        return self.wrapped.config.hidden_size
        
    class Visualizer(Callback):
        def __init__(self,
                     out_dir: pathlib.Path,
                     tokenizer: PreTrainedTokenizerFast):
            self.out_dir = pathlib.Path(out_dir)
            self.tokenizer = tokenizer
            self.special_ids = set(self.tokenizer.all_special_ids)

            self.colors = ["yellow", "fuchsia", "lime", "aqua"]

            self.html_tokens = []

            self.prefix_len = None


        def get_prefix_length(self, ids):
            i = 0
            while i < len(ids) and ids[i] in self.special_ids:
                i += 1
            return i

        def on_predict_epoch_start(self, trainer, pl_module):
            self.html_tokens.clear()
            self.html_tokens.append("<html><body>")

        def on_predict_batch_end(self,
                                 trainer,
                                 pl_module: AggModule,
                                 outputs: AggModule.Output,
                                 batch,
                                 batch_idx,
                                 dataloader_idx = 0):

            stance_enum = pl_module.stance_enum

            condprob_template = "[" + ','.join([f"P({s.name} |" + "{context})" for s in stance_enum]) + "] = {probs}"

            # fullprob_template =   "[" + ','.join([f"P({s.name} | Full Text)" for s in stance_enum]) + "]"
            # aggprob_template =    "[" + ','.join([f"P({s.name} | Aggregated)" for s in stance_enum]) + "]"

            special_ids = self.special_ids
            tokenizer = self.tokenizer

            fullprobs = torch.nn.functional.softmax(outputs.full_logits, dim=-1).cpu().tolist()
            aggprobs = torch.nn.functional.softmax(outputs.agg_logits, dim=-1).cpu().tolist()
            subprobs = torch.nn.functional.softmax(outputs.sub_logits, dim=-1).cpu().tolist()

            batch_size = batch['full']['input_ids'].shape[0]
            subsample_idx = 0
            for sample_idx in range(batch_size):
                id_list = batch['full']['input_ids'][sample_idx].cpu().tolist()
                i = len(id_list) - 1
                while i >= 0 and id_list[i] in special_ids:
                    i -= 1
                target_end = i + 1
                while i >= 0 and id_list[i] not in special_ids:
                    i -= 1
                target_start = i + 1


                target_ids = id_list[target_start:target_end]
                target_str = html.escape(tokenizer.decode(target_ids))
                self.html_tokens.append(f'<p> <strong>Target</strong>: {target_str} </p>')

                if self.prefix_len is None:
                    self.prefix_len = self.get_prefix_length(id_list)
                parent_ids = batch['parent'].tolist()
                context_spans = []

                table_toks = []
                make_prob_cells = lambda probs: "".join([f"<td>{prob:.3f}</td>" for prob in probs])
                table_toks.append("<table>")
                table_toks.append("<thead><tr><th>Source</th>")
                for s in stance_enum:
                    table_toks.append(f"<th>{s.name}</th>")
                table_toks.append("</tr></thead><tbody>")


                color_index = 0
                while subsample_idx < len(parent_ids) and parent_ids[subsample_idx] == sample_idx:
                    subsample_ids = batch['sub']['input_ids'][subsample_idx].cpu().tolist()
                    i = 0
                    while i < len(subsample_ids) and subsample_ids[i] in special_ids:
                        i += 1
                    context_start = i
                    while i < len(subsample_ids) and subsample_ids[i] not in special_ids:
                        i += 1
                    context_end = i
                    decoded = tokenizer.decode(subsample_ids[context_start:context_end])

                    color = self.colors[color_index]
                    highlighted = f'<span style="background-color:{color}">{html.escape(decoded)}</span>'

                    table_toks.append("<tr>")
                    table_toks.append(f'<td style="background-color:{color}">{html.escape(decoded[:15])}&hellip;</td> ')
                    table_toks.append(make_prob_cells(subprobs[subsample_idx]))
                    table_toks.append("</tr>")


                    context_spans.append(highlighted)
                    subsample_idx += 1
                    color_index = (color_index + 1) % len(self.colors)

                context_str = "".join(context_spans)
                table_toks.append(f"<tr><td>Full Context</td>{make_prob_cells(fullprobs[sample_idx])}")
                table_toks.append(f"<tr><td>Recombined  </td>{make_prob_cells( aggprobs[sample_idx])}")
                table_toks.append("</tbody></table>")

                self.html_tokens.append(f'<p> <strong>Context</strong>: {context_str} </p>')
                self.html_tokens.extend(table_toks)



        def on_predict_epoch_end(self, trainer, pl_module):
            self.html_tokens.append("</body></html>")
            html_str = "".join(self.html_tokens)
            with open(self.out_dir / f"highlights.html", 'w') as w:
                w.write(html_str)

        def viz_batch(self, batch: AggModule.Output, metabatch):
            return super().viz_batch(batch, metabatch)