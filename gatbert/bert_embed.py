# STL
import sys
import os
import argparse
import pathlib
# 3rd Party
from tqdm import tqdm
from transformers import BertForPreTraining, BertTokenizerFast
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from torch.utils.data import DataLoader, IterableDataset
import torch
# Local
from .graph import CNGraph, get_entity_embeddings
from .encoder import collate_ids, pretokenize_cn_uri

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=pathlib.Path, required=True, metavar="model_dir/", help="Model directory")
    parser.add_argument("--tok", type=pathlib.Path, metavar="tokenizer_dir/", help="HF tokenizer directory. Defaults to -d")
    parser.add_argument("--model", type=pathlib.Path, metavar="model_dir/", help="HF model directory. Defaults to -d")
    args = parser.parse_args()

    if torch.cuda.is_available():
        pin_memory = True
        pin_memory_device = 'cuda'
        device = torch.device(pin_memory_device)
    else:
        pin_memory = False
        pin_memory_device = None
        device = torch.device('cpu')
        print("CUDA unavailable--running on CPU")

    tok_dir = args.tok or args.d
    model_dir = args.model or args.d

    model = BertForPreTraining.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    uri2id = CNGraph.read_entitites(args.d)
    uri2id = sorted(uri2id.items(), key = lambda pair: pair[1])
    # Ordering check
    uris = []
    for (i, (uri, id)) in enumerate(uri2id):
        assert i == id
        uris.append(uri)

    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(tok_dir)

    class UriDataset(IterableDataset):
        def __init__(self, uris):
            self.__uris = uris
        def __iter__(self):
            map_fn = lambda uri: tokenizer(text=pretokenize_cn_uri(uri), is_split_into_words=True)
            return iter(map(map_fn, self.__uris))

    batch_size = 32
    loader = DataLoader(UriDataset(uris),
                        shuffle=False,
                        batch_size=batch_size,
                        collate_fn=lambda batch: tokenizer.pad(batch, return_tensors='pt'),
                        pin_memory=True)

    uri_embeddings = torch.nn.Embedding(len(uris), model.config.hidden_size, device=device)
    start = 0
    for batch in tqdm(loader, total=len(uris) // batch_size):
        batch = {k:v.to(device) for k,v in batch.items()}
        output = model(**batch, output_hidden_states=True)
        # Last layer, every sample, first token
        current_batch_size = batch['input_ids'].shape[0]
        uri_embeddings.weight.data[start: start + current_batch_size, :] = \
            output.hidden_states[-1][:, 0].detach()
        start += current_batch_size
    torch.save(uri_embeddings, get_entity_embeddings(args.d))
    
