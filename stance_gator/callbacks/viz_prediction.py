import pathlib
import sys
import html
# 3rd Party
import torch
from lightning.pytorch.callbacks import Callback
from transformers import PreTrainedTokenizerFast

class VizPredictionCallback(Callback):

    def __init__(self,
                 out_dir: pathlib.Path,
                 tokenizer: PreTrainedTokenizerFast):
        self.out_dir = pathlib.Path(out_dir)
        self.tokenizer = tokenizer
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.html_tokens = []

    def on_predict_epoch_start(self, trainer, pl_module):
        self.html_tokens.clear()
        self.html_tokens.append("<html>")
        self.html_tokens.append("<head>")
        self.html_tokens.append("<style>")
        self.html_tokens.append(".sample_div { border: 1px solid black; }")
        self.html_tokens.append("table,td,th { border: 1px solid black; }")

        self.html_tokens.append("</style>")
        self.html_tokens.append("</head>")
        self.html_tokens.append("<body>")

    def on_predict_batch_end(self,
                             trainer,
                             pl_module,
                             outputs,
                             batch,
                             batch_idx,
                             dataloader_idx = 0):

        if not all(batch.get(key) is not None for key in ['input_ids']):
            print(f"batch object has no input_ids. Skipping", file=sys.stderr)
            return
        if not all(getattr(outputs, key, None) is not None for key in ['logits']):
            print(f"outputs object has no .logits. Skipping", file=sys.stderr)
            return

        stance_enum = pl_module.stance_enum
        special_ids = self.special_ids
        tokenizer = self.tokenizer



        prob_dists = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().tolist()
        pred_stances = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

        batch_size = batch['input_ids'].shape[0]
        for sample_idx in range(batch_size):
            id_list = batch['input_ids'][sample_idx].cpu().tolist()


            i = len(id_list) - 1
            while i >= 0 and id_list[i] in special_ids:
                i -= 1
            target_end = i + 1
            while i >= 0 and id_list[i] not in special_ids:
                i -= 1
            target_start = i + 1
            target_str = html.escape(tokenizer.decode(id_list[target_start:target_end]))

            i = 0
            while i < target_start and id_list[i] in special_ids:
                i += 1
            context_start = i
            while i < target_start and id_list[i] not in special_ids:
                i += 1
            context_end = i
            context_str = html.escape(tokenizer.decode(id_list[context_start:context_end]))

            label_stance = stance_enum(int(batch['labels'][sample_idx]))
            pred_stance = stance_enum(int(pred_stances[sample_idx]))

            table_toks = []
            table_toks.append("<table>")
            table_toks.append("<thead><tr>")
            for s in stance_enum:
                table_toks.append(f"<th>P({s.name})</th>")
            table_toks.append("</tr></thead><tbody><tr>")
            for i, p in enumerate(prob_dists[sample_idx]):
                if i == pred_stance:
                    cell_col = 'chartreuse' if pred_stance == label_stance else 'red'
                    table_toks.append(f'<td style="background-color:{cell_col}">{p:.3f}</td>')
                else:
                    table_toks.append(f'<td>{p:.3f}</td>')
            table_toks.append('</tbody></table>')


            self.html_tokens.append('<div class="sample_div">')
            self.html_tokens.append(f'<p> <strong>Target</strong>: {target_str} </p>')
            self.html_tokens.append(f'<p> <strong>Context</strong>: {context_str} </p>')
            self.html_tokens.append(f"<p> <strong>Label</strong>: {label_stance.name}")
            self.html_tokens.extend(table_toks)
            self.html_tokens.append('</div>')

    def on_predict_epoch_end(self, trainer, pl_module):
        self.html_tokens.append("</body></html>")
        html_str = "".join(self.html_tokens)
        with open(self.out_dir / f"highlights.html", 'w') as w:
            w.write(html_str)
