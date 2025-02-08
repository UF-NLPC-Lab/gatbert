# 3rd Party
import torch
from transformers import AutoModel, BertModel, AutoConfig
# Local
from .gatbert import GatbertModel, GatbertLayer, GatbertEncoder, GatbertEmbeddings
from .constants import Stance

class GraphClassifier(torch.nn.Module):
    def __init__(self, config, n_relations):
        super().__init__()
        self.bert = GatbertModel(config, n_relations=n_relations)
        self.projection = torch.nn.Linear(
            config.hidden_size,
            out_features=len(Stance),
            bias=False
        )
    def forward(self, *args, **kwargs):
        final_hidden_state = self.bert(*args, **kwargs)
        logits = self.projection(final_hidden_state[:, 0])
        return logits

class ConcatClassifier(torch.nn.Module):
    def __init__(self,
                 pretrained_model_name: str,
                 n_relations: int,
                 n_gat_layers: int = 2):
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.num_hidden_layers = n_gat_layers
        config.n_relations = n_relations

        self.bert = BertModel.from_pretrained(pretrained_model_name)

        self.concept_embeddings = GatbertEmbeddings(config)
        self.concept_embeddings.load_pretrained_weights(self.bert.embeddings)
        self.gat = GatbertEncoder(config)

        self.linear = torch.nn.Linear(2 * config.hidden_size, len(Stance), bias=False)
    
    def forward(self, text, graph):
        # Text Calculation
        bert_out = self.bert(**text)
        text_vec = bert_out.last_hidden_state[:, 0]
        # Graph Calculation
        edge_indices = graph.pop('edge_indices')
        node_counts = graph.pop('node_counts')
        node_counts = torch.maximum(node_counts, torch.tensor(1))
        graph_embeddings = self.concept_embeddings(**graph)
        graph_hidden_states = self.gat(graph_embeddings, edge_indices)
        avg_graph_hidden_states = torch.sum(graph_hidden_states, dim=1) / torch.unsqueeze(node_counts, dim=-1)

        # Concat
        feature_vec = torch.concat([text_vec, avg_graph_hidden_states], dim=-1)

        logits = self.linear(feature_vec)
        return logits

class BertClassifier(torch.nn.Module):
    def __init__(self, pretrained_model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.projection = torch.nn.Linear(
            self.bert.config.hidden_size,
            out_features=len(Stance),
            bias=False
        )
    def forward(self, *args, **kwargs):
        bert_output = self.bert(*args, **kwargs)
        last_hidden_state = bert_output['last_hidden_state'][:, 0]
        logits = self.projection(last_hidden_state)
        return logits