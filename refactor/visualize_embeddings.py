import torch
import numpy as np
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

vocab = Vocabulary().from_files("data/vocabulary")

EMBEDDING_DIM = 128
token_embedding = Embedding(
    num_embeddings=vocab.get_vocab_size("tokens"), embedding_dim=EMBEDDING_DIM
)

with open("saved_models/embeddings.th", "rb") as f:
    token_embedding.load_state_dict(torch.load(f))

token_ids = torch.tensor([x for x in range(2, vocab.get_vocab_size())], dtype=torch.long)

token_embedding.eval()
with torch.no_grad():
    embeddings = token_embedding(token_ids).cpu().numpy()

print(embeddings.shape)
tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=300)
tsne_results = tsne.fit_transform(embeddings)

plt.figure()
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=1)
for i, (x, y) in enumerate(tsne_results):
    plt.text(x, y, vocab.get_token_from_index(i+2), fontsize=2)
plt.savefig("tsne_results.pdf")

print(embeddings.shape)
pca = PCA(n_components=2)
pca_results = pca.fit_transform(embeddings)

plt.figure()
plt.scatter(pca_results[:, 0], pca_results[:, 1], s=1)
for i, (x, y) in enumerate(pca_results):
    plt.text(x, y, vocab.get_token_from_index(i+2), fontsize=2)
plt.savefig("pca_results.pdf")