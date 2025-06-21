# STL
from collections import namedtuple
# 3rd Party
import torch
from transformers import BertTokenizerFast, BertModel
# Local
from ..data import Encoder, Sample, encode_text, collate_ids, keyed_scalar_stack
from ..constants import DEFAULT_MODEL
from .base_module import StanceModule

class RnnModule(StanceModule):

    Output = namedtuple("Output", ["logits", "loss", "dist_history"])

    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL,
                 **parent_kwargs,
                 ):
        super().__init__(**parent_kwargs)
        loaded_model = BertModel.from_pretrained(pretrained_model)
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
        self.pad_token = self.tokenizer.pad_token_id
        self.num_labels = len(self.stance_enum)

        self.word_embeddings = loaded_model.embeddings.word_embeddings
        hidden_size = self.word_embeddings.embedding_dim
        self.hidden_size = hidden_size
        self.cell = torch.nn.LSTMCell(hidden_size, hidden_size)

        self.word_kproj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_kproj    = torch.nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.rel_fnn = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Flatten(start_dim=-2, end_dim=-1)
        )
        self.classifier = self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.num_labels, bias=True)
        )
        self.__encoder = RnnModule.Encoder(self.tokenizer)

    @property
    def encoder(self):
        return self.__encoder

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size, seq_length = input_ids.shape[:2]
        embedded = self.word_embeddings(input_ids)

        # Start with uniform distro
        dist_prev = torch.ones((batch_size, self.num_labels), dtype=embedded.dtype, device=embedded.device) / self.num_labels
        h_prev = torch.zeros((batch_size, self.hidden_size), dtype=embedded.dtype, device=embedded.device)
        c_prev = torch.zeros((batch_size, self.hidden_size), dtype=embedded.dtype, device=embedded.device)

        dist_history = []

        for t in range(seq_length):
            inputs_t = embedded[:, t]

            h_t, c_t = self.cell(inputs_t, (h_prev, c_prev))

            rel_prev = self.rel_fnn(h_prev)
            rel_t = self.rel_fnn(h_t) 
            rel_logits = torch.stack([rel_prev, rel_t], dim=-1)
            rel_dist = torch.nn.functional.softmax(rel_logits, dim=-1)

            dist_hat = torch.nn.functional.softmax(self.classifier(h_t), dim=-1)
            dist_t = dist_prev * rel_dist[:, [0]] + dist_hat * rel_dist[:, [1]]
            dist_history.append(dist_t)

            dist_prev = dist_t
            h_prev = h_t
            c_prev = c_t

        dist_history = torch.stack(dist_history, dim=1)
        sample_lengths = torch.sum(attention_mask, dim=-1)
        batch_inds = torch.arange(0, batch_size, device=dist_history.device)
        final_dists = dist_history[batch_inds, sample_lengths - 1]
        log_probs = torch.log(final_dists)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.nll_loss(log_probs, labels)
        return self.Output(logits=log_probs, loss=loss, dist_history=dist_history)


    class Encoder(Encoder):
        def __init__(self, tokenizer: BertTokenizerFast):
            self.tokenizer = tokenizer

        def encode(self, sample: Sample):
            encoded = encode_text(self.tokenizer, sample)
            encoded.pop('special_tokens_mask', None)
            encoded.pop('token_type_ids', None)
            encoded['labels'] = torch.tensor([sample.stance.value])
            return encoded
        def collate(self, samples):
            collated = collate_ids(self.tokenizer,
                               samples,
                               return_attention_mask=True)
            collated['labels'] = keyed_scalar_stack(samples, 'labels')
            return collated