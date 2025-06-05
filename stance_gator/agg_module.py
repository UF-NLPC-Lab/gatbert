# STL
from __future__ import annotations
from typing import Optional, Literal, List
from itertools import product
import dataclasses
import pathlib
import html
import enum
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
from .dep_tools import get_spans

class AggModule(StanceModule):

    @dataclasses.dataclass
    class Output:
        full_logits: torch.Tensor
        full_features: torch.Tensor
        agg_logits: Optional[torch.Tensor] = None
        sub_logits: Optional[torch.Tensor] = None

        @property
        def logits(self):
            return self.agg_logits if self.agg_logits is not None else self.full_logits

    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL,
                 classifier_hidden_units: Optional[int] = None,
                 cont_loss: bool = False,
                 cl_temp: float = 1.,
                 cl_weight: float = 1.,
                 subsamples: bool = False,
                 subsample_cont_loss: bool = False,
                 agg: Literal['mean'] = 'mean',
                 **parent_kwargs,
                 ):
        super().__init__(**parent_kwargs)
        self.save_hyperparameters()
        self.cont_loss = cont_loss
        self.cl_temp = cl_temp
        self.cl_weight = cl_weight
        self.subsamples = subsamples
        self.subsample_cont_loss = subsample_cont_loss
        self.agg = agg
        config = BertForStanceConfig.from_pretrained(pretrained_model,
                                                     classifier_hidden_units=classifier_hidden_units,
                                                     id2label=self.stance_enum.id2label(),
                                                     label2id=self.stance_enum.label2id(),
                                                     )

        self.wrapped = BertForStance.from_pretrained(pretrained_model, config=config)
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(pretrained_model)
        self.__encoder = AggModule.Encoder(self)

        hidden_size = config.hidden_size
        self.span_ffn = torch.nn.Sequential(
            torch.nn.Dropout(config.hidden_dropout_prob),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 2, bias=True),

        )

        if self.cont_loss:
            self.sim_proj = torch.nn.Linear(1, 1)

    def _eval_step(self, batch, batch_idx, stage):
        batch.pop('cl_labels', None)
        return super()._eval_step(batch, batch_idx, stage)

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        cl_labels = batch.pop("cl_labels", None)
        output_obj: AggModule.Output = self(**batch)

        ce_full = torch.nn.functional.cross_entropy(output_obj.full_logits, labels)
        self.log("train_ce_full", ce_full)

        total_loss = ce_full

        if cl_labels is not None:
            normalized_vecs = output_obj.full_features / torch.norm(output_obj.full_features, p=2, dim=-1, keepdim=True)
            cosine_sims = normalized_vecs @ normalized_vecs.transpose(1, 0)
            sim_logits = self.sim_proj(cosine_sims.view(-1, 1))
            cont_loss_val = torch.nn.functional.binary_cross_entropy_with_logits(sim_logits.flatten(), cl_labels.flatten().to(sim_logits.dtype))

            # scaled_sims = torch.exp(cosine_sims / self.cl_temp)
            # diag_mask = torch.eye(scaled_sims.shape[0], dtype=torch.bool, device=scaled_sims.device)
            # scaled_sims = torch.where(diag_mask, 0, scaled_sims)
            # pos_pairs = torch.where(cl_labels, scaled_sims, 0.)
            # log_probs = torch.log(torch.sum(pos_pairs, -1) / torch.sum(scaled_sims, -1))
            # cont_loss_val = -torch.mean(log_probs)

            self.log("train_cl", cont_loss_val)
            total_loss += self.cl_weight * cont_loss_val

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
        full_features = full_pass.last_hidden_state[:, 0]

        sub_log = None
        agg_log = None
        if sub is not None:
            assert parent is not None
            assert self.agg == 'mean'
            sub_pass: BertForStance.Output = self.wrapped(**sub)
            sub_probs = torch.nn.functional.softmax(sub_pass.logits, dim=-1)
            agg_probs = torch_scatter.scatter(sub_probs, parent, dim=-2, reduce='mean')
            agg_log = torch.log(agg_probs) # TODO: Do something more numerically stable
            sub_log = sub_pass.logits
        return AggModule.Output(full_logits=full_pass.logits,
                                full_features=full_features,
                                agg_logits=agg_log,
                                sub_logits=sub_log)
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
            meta = span_indices
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

            if self.module.cont_loss:
                rdict['cl_labels'] = torch.unsqueeze(labels, -1) == torch.unsqueeze(labels, 0)

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
            required = [outputs.sub_logits, outputs.agg_logits, batch.get('sub'), batch.get('parent')]
            if not all(x is not None for x in required):
                return
            stance_enum = pl_module.stance_enum
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

                parent_ids = batch['parent'].tolist()
                context_spans = []

                table_toks = []
                make_prob_cells = lambda probs: "".join([f"<td>{prob:.3f}</td>" for prob in probs])
                table_toks.append("<table>")
                table_toks.append("<thead><tr><th>Source</th>")
                for s in stance_enum:
                    table_toks.append(f"<th>P({s.name})</th>")
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
                    highlighted = f'<span style="background-color:{color}">{html.escape(decoded)}&nbsp;</span>'

                    table_toks.append("<tr>")
                    table_toks.append(f'<td style="background-color:{color}">{html.escape(decoded[:15])}&hellip;</td> ')
                    table_toks.append(make_prob_cells(subprobs[subsample_idx]))
                    table_toks.append("</tr>")


                    context_spans.append(highlighted)
                    subsample_idx += 1
                    color_index = (color_index + 1) % len(self.colors)

                table_toks.append(f"<tr><td>Full Context</td>{make_prob_cells(fullprobs[sample_idx])}")
                table_toks.append(f"<tr><td>Recombined  </td>{make_prob_cells( aggprobs[sample_idx])}")
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