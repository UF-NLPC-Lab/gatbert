from collections import namedtuple
# 3rd Party
import torch
from typing import Optional
from transformers import BertModel, BertTokenizerFast
# Local
from .base_module import StanceModule
from .constants import DEFAULT_MODEL, Stance
from .encoder import SimpleEncoder

class BsrgcnModule(StanceModule):
    def __init__(self,
                graph_dim: int = 100,
                pretrained_model = DEFAULT_MODEL,
                ):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        config = self.bert.config
        hidden_size = config.hidden_size

        self.classifier = torch.nn.Sequential(
            # torch.nn.Dropout(config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob),
            # torch.nn.Linear(hidden_size, classifier_hidden_units, bias=True),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden_size + graph_dim, len(Stance), bias=True)
        )
        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.graph_enc_ffn = torch.nn.Sequential(
            torch.nn.Linear(graph_dim, graph_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        )
        self.graph_dec_ffn = torch.nn.Sequential(
            torch.nn.Linear(graph_dim, graph_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(graph_dim, graph_dim)
        )
        self.ce_func = torch.nn.CrossEntropyLoss()
        self.mse_func = torch.nn.MSELoss()

        self.__encoder = SimpleEncoder(
            BertTokenizerFast.from_pretrained(pretrained_model)
        )

    @property
    def encoder(self):
        return self.__encoder

    Output = namedtuple("BsrgcnOutput", field_names=["logits"])

    def training_step(self, batch, batch_idx):
        graph_embeds = batch.pop('graph_embeds')
        labels       = batch.pop('labels')

        bert_output = self.bert(**batch)
        cls_hidden_state = bert_output[0][:, 0]

        graph_enc = self.graph_enc_ffn(graph_embeds)
        feature_vec = torch.cat([cls_hidden_state, graph_enc], dim=1)
        logits = self.classifier(feature_vec)
        stance_loss = self.ce_func(logits, labels)
        graph_recon = self.graph_dec_ffn(graph_enc)
        recon_loss = self.mse_func(graph_recon, graph_embeds)
        loss = stance_loss + recon_loss
        self.log("loss", loss)
        self.log("loss/stance", stance_loss)
        self.log("loss/recon", recon_loss)
        return loss

    def forward(self, graph_embeds, *bert_args, **bert_kwargs):
        bert_output = self.bert(*bert_args, **bert_kwargs)
        cls_hidden_state = bert_output[0][:, 0]
        graph_enc = self.graph_enc_ffn(graph_embeds)
        feature_vec = torch.cat([cls_hidden_state, graph_enc], dim=1)
        logits = self.classifier(feature_vec)
        return BsrgcnModule.Output(logits)