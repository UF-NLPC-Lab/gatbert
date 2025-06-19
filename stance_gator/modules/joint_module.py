from collections import namedtuple
# 3rd Party
import pathlib
import torch
from transformers import BertModel, BertTokenizerFast
import torch_geometric.data
# Local
from .base_module import StanceModule
from ..constants import DEFAULT_MODEL
from ..data import SimpleEncoder, Encoder, keyed_scalar_stack, CN
from ..rgcn import CNEncoder, text2edges, make_graph_sample

class JointModule(StanceModule):
    def __init__(self,
                cn_path: pathlib.Path,
                graph_dim: int = 256,
                pretrained_model = DEFAULT_MODEL,
                warmup_epochs: int = 5,
                mask_rate: float = 0.75,
                **parent_kwargs
                ):
        super().__init__(**parent_kwargs)

        self.rgcn = CNEncoder(cn_path, dim=graph_dim)
        self.cn = self.rgcn.cn
        self.mask_rate = mask_rate

        self.bert = BertModel.from_pretrained(pretrained_model)
        config = self.bert.config

        hidden_size = config.hidden_size
        feature_size = hidden_size + graph_dim
        self.hidden_size = hidden_size

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob),
            torch.nn.Linear(feature_size, feature_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_size, len(self.stance_enum), bias=True)
        )
        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.graph_enc_ffn = torch.nn.Sequential(
            torch.nn.Linear(graph_dim, graph_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        )
        self.ce_func = torch.nn.CrossEntropyLoss()


        self.__encoder = self.Encoder(
            self.cn,
            SimpleEncoder(BertTokenizerFast.from_pretrained(pretrained_model))
        )
        self.__warmup_epochs = warmup_epochs

        self.__bert_freeze = set()
        self.bert.train()
        for name, p in self.bert.named_parameters():
            if p.requires_grad_:
                self.__bert_freeze.add(name)
                p.requires_grad_ = False
        self.__frozen = True

    @property
    def encoder(self):
        return self.__encoder

    Output = namedtuple("JointOutput", field_names=["logits", "loss"])

    def on_train_epoch_start(self):
        if self.__frozen and self.current_epoch >= self.__warmup_epochs:
            for name, p in self.bert.named_parameters():
                if name in self.__bert_freeze:
                    p.requires_grad_ = True
            self.__frozen = False

    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.__warmup_epochs:
            # Randomly mask some of the hidden units from BERT 
            mask = torch.rand((1, self.hidden_size), device=self.device) > self.mask_rate
            batch['mask'] = mask
        return super().training_step(batch, batch_idx)

    def forward(self, text, graph, labels=None, mask=None):
        bert_output = self.bert(**text)
        cls_hidden_state = bert_output[0][:, 0]

        if mask is not None:
            cls_hidden_state = cls_hidden_state * mask

        graph_output = self.rgcn(graph)
        graph_enc = self.graph_enc_ffn(graph_output)

        feature_vec = torch.cat([cls_hidden_state, graph_enc], dim=1)
        logits = self.classifier(feature_vec)

        loss = None
        if labels is not None:
            loss = self.ce_func(logits, labels)
        return self.Output(logits, loss)

    class Encoder(Encoder):
        def __init__(self, cn: CN, nested: SimpleEncoder):
            self.cn = cn
            self.nested = nested
            pass

        def encode(self, sample):
            text_encoding = self.nested.encode(sample)
            edges = text2edges(self.cn, sample)
            graph_encoding = make_graph_sample(edges, n_relations=len(self.cn.relation2id))
            rdict = {}
            rdict['labels'] = text_encoding.pop('labels')
            rdict['text'] = text_encoding
            rdict['graph'] = graph_encoding
            return rdict
        
        def collate(self, samples):
            graph_batch = torch_geometric.data.Batch.from_data_list([s['graph'] for s in samples])
            text_batch = self.nested.collate([s['text'] for s in samples])
            rdict = {
                "text": text_batch,
                "graph": graph_batch,
                "labels": keyed_scalar_stack(samples, 'labels')
            }
            return rdict

