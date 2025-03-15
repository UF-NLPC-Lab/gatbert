# STL
import os
import argparse
import pathlib
# 3rd Party
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizerFast
from torch.utils.data import DataLoader, IterableDataset
import torch
# Local
from .graph import CNGraph, get_entity_embeddings
from .encoder import collate_ids, pretokenize_cn_uri

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=pathlib.Path, required=True, metavar="model_dir/", help="Model directory")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("CUDA unavailable--running on CPU")

    model = BertForMaskedLM.from_pretrained(args.d)
    model.eval()
    model.to(device)

    uri2id = CNGraph.read_entitites(args.d)
    uri2id = sorted(uri2id.items(), key = lambda pair: pair[1])
    # Ordering check
    uris = []
    for (i, (uri, id)) in enumerate(uri2id):
        assert i == id
        uris.append(uri)

    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(args.d)
    uri_embeddings = torch.nn.Embedding(len(uris), model.config.hidden_size, device=device)

    cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    num_workers = cpus - 1

    class UriDataset(IterableDataset):
        def __init__(self, uris):
            self.__uris = uris
        def __iter__(self):
            def map_fn(uri: str):
                encoding = tokenizer(text=pretokenize_cn_uri(uri), is_split_into_words=True, return_tensors='pt')
                return {k:v.to(device) for k,v in encoding.items()}
            return iter(map(map_fn, self.__uris))

    batch_size = 32
    loader = DataLoader(UriDataset(uris),
                        shuffle=False,
                        batch_size=batch_size,
                        collate_fn=lambda batch: collate_ids(tokenizer, batch, return_attention_mask=True))

    start = 0
    for batch in tqdm(loader, total=len(uris) // batch_size):
        output = model(**batch, output_hidden_states=True)
        # Last layer, every sample, first token
        current_batch_size = batch['input_ids'].shape[0]
        uri_embeddings.weight.data[start: start + current_batch_size, :] = \
            output.hidden_states[-1][:, 0].detach()
        start += current_batch_size
    torch.save(uri_embeddings, get_entity_embeddings(args.d))
    
