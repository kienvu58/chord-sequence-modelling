import torch
import numpy as np
from modules.data_preprocessors import parse_chord_name, get_key_number
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


def label_to_color_and_marker(label):
    key, form_stuff, _ = parse_chord_name(label)
    if key is None:
        return "black", "."
    key_number = get_key_number(key)
    color_list = [
        "red",
        "maroon",
        "yellow",
        "olive",
        "lime",
        "green",
        "aqua",
        "teal",
        "blue",
        "navy",
        "fuchsia",
        "purple",
    ]

    form_list = ["G", "I", "F", "M", "+", "o", "%", "m", None]
    form = form_stuff[0]
    try:
        form_index = form_list.index(form)
    except Exception as e:
        print(e)
        print(label)

    marker_list = [".", ".", ".", "v", "p", "D", "x", "*", "s"]

    return color_list[key_number], marker_list[form_index]


vocab = Vocabulary().from_files("data/vocabulary")

EMBEDDING_DIM = 128
token_embedding = Embedding(
    num_embeddings=vocab.get_vocab_size("tokens"), embedding_dim=EMBEDDING_DIM
)

token_embedding.load_state_dict(torch.load("saved_models/word2vec.th"))


token_ids = torch.tensor(
    [x for x in range(2, vocab.get_vocab_size())], dtype=torch.long
)

if torch.cuda.is_available():
    cuda_device = 0
    token_embedding = token_embedding.cuda(cuda_device)
    token_ids = token_ids.cuda(cuda_device)
else:
    cuda_device = -1

token_embedding.eval()
with torch.no_grad():
    embeddings = token_embedding(token_ids).cpu().numpy()

print(embeddings.shape)
tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=300)
tsne_results = tsne.fit_transform(embeddings)

plt.figure()
for i, (x, y) in enumerate(tsne_results):
    label = vocab.get_token_from_index(i + 2)
    color, marker = label_to_color_and_marker(label)
    plt.scatter(x, y, s=1, c=color, marker=marker)
    plt.text(x, y, label, fontsize=2, color=color)
plt.savefig("figures/tsne_results.pdf")

print(embeddings.shape)
pca = PCA(n_components=2)
pca_results = pca.fit_transform(embeddings)

plt.figure()
for i, (x, y) in enumerate(pca_results):
    label = vocab.get_token_from_index(i + 2)
    color, marker = label_to_color_and_marker(label)
    plt.scatter(x, y, s=1, c=color, marker=marker)
    plt.text(x, y, label, fontsize=2, color=color)
plt.savefig("figures/pca_results.pdf")
