# STL
from __future__ import annotations
from typing import Optional, Literal, List
import logging
import dataclasses
import pathlib
import html
# 3rd Party
from lightning.pytorch.callbacks import Callback
import torch_scatter
import torch
from transformers import BertTokenizerFast, PreTrainedTokenizerFast
# Local
from ..types import TensorDict
from ..data import SPACY_PIPES
from ..sample import Sample
from ..models import BertForStance, BertForStanceConfig
from ..encoder import SimpleEncoder, Encoder, keyed_scalar_stack
from ..constants import DEFAULT_MODEL
from .base_module import StanceModule
from ..dep_tools import get_spans

class AggModule(StanceModule):

    LOGGER = logging.getLogger("AggModule")

    @dataclasses.dataclass
    class Output:
        full_logits: torch.Tensor
        agg_logits: Optional[torch.Tensor] = None
        sub_logits: Optional[torch.Tensor] = None
        att_weights: Optional[torch.Tensor] = None

        @property
        def logits(self):
            return self.agg_logits if self.agg_logits is not None else self.full_logits

    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL,
                 classifier_hidden_units: Optional[int] = None,
                 subsamples: bool = False,
                 agg: Literal['mean', 'att'] = 'mean',
                 **parent_kwargs,
                 ):
        super().__init__(**parent_kwargs)
        self.save_hyperparameters()

        self.subsamples = subsamples
        self.agg = agg
        config = BertForStanceConfig.from_pretrained(pretrained_model,
                                                     classifier_hidden_units=classifier_hidden_units,
                                                     id2label=self.stance_enum.id2label(),
                                                     label2id=self.stance_enum.label2id(),
                                                     )

        self.wrapped = BertForStance.from_pretrained(pretrained_model, config=config)
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(pretrained_model)
        self.__encoder = AggModule.Encoder(self)

        if self.agg == 'att':
            self.att_query = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, 1, bias=False),
                torch.nn.Flatten(start_dim=-2, end_dim=-1)
            )

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        output_obj: AggModule.Output = self(**batch)

        ce_full = torch.nn.functional.cross_entropy(output_obj.full_logits, labels)
        self.log("train_ce_full", ce_full)

        total_loss = ce_full

        if output_obj.agg_logits is not None:
            ce_agg  = torch.nn.functional.cross_entropy(output_obj.agg_logits, labels)
            self.log("train_ce_agg", ce_agg)
            total_loss += ce_agg

        self.log("train_loss", total_loss)
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        sub_batch = {k:v for k,v in batch.items() if 'labels' not in k}
        return super().predict_step(sub_batch, batch_idx, dataloader_idx)

    def forward(self, full, sub=None, parent=None):
        full_pass: BertForStance.Output = self.wrapped(**full)
        sub_log = None
        agg_log = None
        att_weights = None
        if sub is not None:
            assert parent is not None
            sub_pass: BertForStance.Output = self.wrapped(**sub)
            sub_probs = torch.nn.functional.softmax(sub_pass.logits, dim=-1)
            if self.agg == 'mean':
                agg_probs = torch_scatter.scatter(sub_probs, parent, dim=-2, reduce='mean')
                parent_counts = torch_scatter.scatter(torch.ones_like(parent), parent, reduce='sum')
                att_weights = 1. / parent_counts[parent]
            else:
                assert self.agg == 'att'
                att_logits = self.att_query(sub_pass.last_hidden_state[:, 0])
                exp_att_logits = torch.exp(att_logits)
                norm_constants = torch_scatter.scatter(exp_att_logits, parent, dim=-1, reduce='sum')
                att_weights = exp_att_logits / norm_constants[parent]
                weighted_sub_probs = sub_probs * torch.unsqueeze(att_weights, -1)
                agg_probs = torch_scatter.scatter(weighted_sub_probs, parent, dim=-2, reduce='sum')
            agg_log = torch.log(agg_probs) # TODO: Do something more numerically stable
            sub_log = sub_pass.logits
        return AggModule.Output(full_logits=full_pass.logits,
                                agg_logits=agg_log,
                                sub_logits=sub_log,
                                att_weights=att_weights)
    @property
    def encoder(self) -> AggModule.Encoder:
        return self.__encoder

    def make_visualizer(self, output_dir):
        return AggModule.Visualizer(output_dir, self.tokenizer)

    class Encoder(Encoder):
        def __init__(self, module: AggModule):
            self.module = module
            self.tokenizer = module.tokenizer
            self.nested = SimpleEncoder(self.tokenizer, max_context_length=None, max_target_length=None)

        def encode(self, sample: Sample):
            encode_dict = {}

            full_context = " ".join(sample.context) if sample.is_split_into_words else sample.context
            target = " ".join(sample.target) if sample.is_split_into_words else sample.target
            sample = Sample(context=full_context,
                            target=target,
                            is_split_into_words=False,
                            lang=sample.lang,
                            stance=sample.stance)
            full_encoding = self.nested.encode(sample)
            encode_dict["labels"] = full_encoding.pop('labels')
            encode_dict["full"] = full_encoding

            if not self.module.subsamples:
                return encode_dict

            lang = sample.lang or 'en'
            pipeline = SPACY_PIPES[lang]

            span_indices = []
            spacy_doc = pipeline(full_context)

            subsample_encodings = []
            for (start, end) in get_spans(spacy_doc):
                sub_sample = Sample(context=str(spacy_doc[start:end]),
                                    target=target,
                                    stance=sample.stance,
                                    is_split_into_words=False,
                                    lang=lang)
                subsample_encoding = self.nested.encode(sub_sample)
                subsample_encoding.pop('labels')
                subsample_encodings.append(subsample_encoding)
                span_indices.append((start, end))
            encode_dict["sub"] = subsample_encodings
            return encode_dict

        def collate(self, samples: List[TensorDict]):
            rdict = {}

            if "sub" in samples[0]:
                parent_inds = []
                for i, sample in enumerate(samples):
                    num_subsamples = len(sample['sub'])
                    parent_inds.extend(i for _ in range(num_subsamples))
                parent_inds = torch.tensor(parent_inds)
                subsample_batch = self.nested.collate([sub_samp for s in samples for sub_samp in s['sub']])
                rdict.update({"parent": parent_inds, "sub": subsample_batch})

            full_batch = self.nested.collate([s['full'] for s in samples])
            labels = keyed_scalar_stack(samples, 'labels')
            rdict.update({
                "full": full_batch,
                "labels": labels,
            })

            return rdict


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


        def get_prefix_length(self, ids):
            i = 0
            while i < len(ids) and ids[i] in self.special_ids:
                i += 1
            return i

        def on_predict_epoch_start(self, trainer, pl_module):
            self.html_tokens.clear()
            self.html_tokens.append("<html>")
            self.html_tokens.append("<head><style> .sample_div { border: 1px solid black; } </style></head>")
            self.html_tokens.append("<body>")

        def on_predict_batch_end(self,
                                 trainer,
                                 pl_module: AggModule,
                                 outputs: AggModule.Output,
                                 batch,
                                 batch_idx,
                                 dataloader_idx = 0):
            required = [outputs.sub_logits, outputs.agg_logits, outputs.att_weights, batch.get('sub'), batch.get('parent')]
            if not all(x is not None for x in required):
                return
            stance_enum = pl_module.stance_enum
            special_ids = self.special_ids
            tokenizer = self.tokenizer

            fullprobs = torch.nn.functional.softmax(outputs.full_logits, dim=-1).cpu().tolist()
            aggprobs = torch.nn.functional.softmax(outputs.agg_logits, dim=-1).cpu().tolist()
            subprobs = torch.nn.functional.softmax(outputs.sub_logits, dim=-1).cpu().tolist()
            att_weights = outputs.att_weights.cpu().tolist()

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

                parent_ids = batch['parent'].tolist()
                context_spans = []

                table_toks = []
                make_prob_cells = lambda probs: "".join([f"<td>{prob:.3f}</td>" for prob in probs])
                table_toks.append("<table>")
                table_toks.append("<thead><tr><th>Source</th>")
                for s in stance_enum:
                    table_toks.append(f"<th>P({s.name})</th>")
                table_toks.append("<th>Att. Weight</th></tr></thead><tbody>")


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
                    highlighted = f'<span style="background-color:{color}">{html.escape(decoded)}&nbsp;</span>'

                    table_toks.append("<tr>")
                    table_toks.append(f'<td style="background-color:{color}">{html.escape(decoded[:15])}&hellip;</td> ')
                    table_toks.append(make_prob_cells(subprobs[subsample_idx]))
                    table_toks.append(f'<td>{att_weights[subsample_idx]:.3f}</td>')
                    table_toks.append("</tr>")


                    context_spans.append(highlighted)
                    subsample_idx += 1
                    color_index = (color_index + 1) % len(self.colors)

                table_toks.append(f"<tr><td>Full Context</td>{make_prob_cells(fullprobs[sample_idx])}<td></td></tr>")
                table_toks.append(f"<tr><td>Recombined  </td>{make_prob_cells( aggprobs[sample_idx])}<td></td></tr>")
                table_toks.append("</tbody></table>")

                self.html_tokens.append('<div class="sample_div">')
                self.html_tokens.append(f'<p> <strong>Target</strong>: {target_str} </p>')
                context_str = "".join(context_spans)
                self.html_tokens.append(f'<p> <strong>Context</strong>: {context_str} </p>')
                self.html_tokens.append(f"<p> <strong>Label</strong>: {stance_enum(int(batch['labels'][sample_idx])).name}")
                self.html_tokens.extend(table_toks)
                self.html_tokens.append('</div>')




        def on_predict_epoch_end(self, trainer, pl_module):
            self.html_tokens.append("</body></html>")
            html_str = "".join(self.html_tokens)
            with open(self.out_dir / f"highlights.html", 'w') as w:
                w.write(html_str)

        def viz_batch(self, batch: AggModule.Output, metabatch):
            return super().viz_batch(batch, metabatch)