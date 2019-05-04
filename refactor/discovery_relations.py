import torch
import numpy as np
from allennlp.data.vocabulary import Vocabulary
from torch.nn import CosineSimilarity
from collections import Counter
from allennlp.modules.token_embedders import Embedding

def discovery(embedding, vocab, chord_a, chord_b, chord_c, num_output=10):
    a_id = vocab.get_token_index(chord_a)
    b_id = vocab.get_token_index(chord_b)
    c_id = vocab.get_token_index(chord_c)
    vec_a = embedding.weight[a_id]
    vec_b = embedding.weight[b_id]
    vec_c = embedding.weight[c_id]
    cosine = CosineSimilarity(dim=0)
    sims = Counter()

    vec = vec_b - vec_a + vec_c

    for index, token in vocab.get_index_to_token_vocabulary().items():
        sim = cosine(vec, embedding.weight[index]).item()
        sims[token] = sim

    return sims.most_common(num_output)

vocab = Vocabulary().from_files("data/vocabulary")

EMBEDDING_DIM = 128
token_embedding = Embedding(
    num_embeddings=vocab.get_vocab_size("tokens"), embedding_dim=EMBEDDING_DIM
)

with open("saved_models/word2vec.th", "rb") as f:
    token_embedding.load_state_dict(torch.load(f, map_location="cpu"))

print(discovery(token_embedding, vocab, "C", "G", "G"))
