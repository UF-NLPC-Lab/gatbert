import abc
# 3rd Party
import torch
from transformers import AutoModel, BertModel, AutoConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertConfig
# Local
from .gatbert import GatbertModel, GatbertLayer, GatbertEncoder, GatbertEmbeddings
from .constants import Stance
from .config import GatbertConfig

class StanceClassifier(torch.nn.Module):
    def __init__(self, config: GatbertConfig):
        super().__init__()
        self.config = config
        pass

    @abc.abstractmethod
    def load_pretrained_weights(self):
        pass

class GraphOnlyClassifier(StanceClassifier):
    """
    Produces a hidden state summarizing just a graph (no text)
    """
    def __init__(self, config: GatbertConfig):
        super().__init__(config)
        self.concept_embeddings = GatbertEmbeddings(config)
        self.gat = GatbertEncoder(config)
        self.linear = torch.nn.Linear(config.hidden_size, len(Stance), bias=False)

    def load_pretrained_weights(self):
        self.concept_embeddings.load_pretrained_weights(BertModel.from_pretrained(self.config.base_model).embeddings)

    def forward(self, input_ids, pooling_mask, edge_indices, node_counts):
        # Graph Calculation
        graph_embeddings = self.concept_embeddings(input_ids=input_ids,
                                                   pooling_mask=pooling_mask)
        graph_hidden_states = self.gat(graph_embeddings, edge_indices)
        node_counts = torch.maximum(node_counts, torch.tensor(1))
        avg_graph_hidden_states = torch.sum(graph_hidden_states, dim=1) / torch.unsqueeze(node_counts, dim=-1)
        logits = self.linear(avg_graph_hidden_states)
        return logits

class GraphClassifier(StanceClassifier):
    def __init__(self, config: GatbertConfig):
        super().__init__(config)
        self.bert = GatbertModel(config)
        self.projection = torch.nn.Linear(
            config.hidden_size,
            out_features=len(Stance),
            bias=False
        )

    def load_pretrained_weights(self):
        self.bert.load_pretrained_weights(BertModel.from_pretrained(self.config.base_model))

    def forward(self, *args, **kwargs):
        final_hidden_state = self.bert(*args, **kwargs)
        logits = self.projection(final_hidden_state[:, 0])
        return logits

class ConcatClassifier(StanceClassifier):
    def __init__(self, config: GatbertConfig):
        """
        Args:
            pretrained_model_name: model to load for text portion of the model
            config: config for the graph portion of the model
        """
        super().__init__(config)

        self.bert = BertModel.from_pretrained(config.base_model)
        self.concept_embeddings = GatbertEmbeddings(config)
        self.gat = GatbertEncoder(config)

        self.linear = torch.nn.Linear(config.hidden_size + self.bert.config.hidden_size, len(Stance), bias=False)
    
    def load_pretrained_weights(self):
        self.concept_embeddings.load_pretrained_weights(BertModel.from_pretrained(self.config.base_model).embeddings)

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

class BertClassifier(StanceClassifier):
    def __init__(self, config: GatbertConfig):
        super().__init__(config)
        self.bert = BertModel( BertConfig.from_pretrained(config.base_model) )
        self.projection = torch.nn.Linear(
            self.bert.config.hidden_size,
            out_features=len(Stance),
            bias=False
        )

    def load_pretrained_weights(self):
        # TODO: Find a cleaner way than just reinstantiating the object
        self.bert = BertModel.from_pretrained(self.config.base_model)

    def forward(self, *args, **kwargs):
        bert_output = self.bert(*args, **kwargs)
        last_hidden_state = bert_output['last_hidden_state'][:, 0]
        logits = self.projection(last_hidden_state)
        return logits