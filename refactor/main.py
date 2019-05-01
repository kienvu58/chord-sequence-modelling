import torch
import torch.optim as optim
import numpy as np

from modules.dataset_readers import CpmDatasetReader
from modules.chord_progression_models import Cpm
from modules.predictors import Predictor

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import (
    TextFieldEmbedder,
    BasicTextFieldEmbedder,
)
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import (
    Seq2SeqEncoder,
    PytorchSeq2SeqWrapper,
    IntraSentenceAttentionEncoder,
    StackedSelfAttentionEncoder,
)
from allennlp.modules.similarity_functions import MultiHeadedSimilarity
from allennlp.common.file_utils import cached_path
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

torch.manual_seed(1)


def train(
    train_dataset,
    val_dataset,
    test_dataset,
    vocab,
    word_embedder,
    contextualizer,
    hparams,
    contextual_embedding_dropout=None,
):
    model = Cpm(vocab, word_embedder, contextualizer)
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    print(cuda_device)

    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])

    iterator = BucketIterator(
        batch_size=hparams["batch_size"], sorting_keys=[("input_tokens", "num_tokens")]
    )
    iterator.index_with(vocab)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        patience=10,
        num_epochs=hparams["num_epochs"],
        cuda_device=cuda_device,
    )
    trainer.train()
    model_saved_path = "saved_models/tmp.th"
    with open(model_saved_path, "wb") as f:
        torch.save(model.state_dict(), f)

    predictor = Predictor(model=model, iterator=iterator, cuda_device=cuda_device)
    pred_metrics = predictor.predict(test_dataset)
    return pred_metrics, model_saved_path


reader = CpmDatasetReader()
train_dataset = reader.read("data/cv/1/train.txt")
val_dataset = reader.read("data/cv/1/val.txt")
test_dataset = reader.read("data/cv/1/test.txt")

vocab = Vocabulary().from_files("data/vocabulary")

EMBEDDING_DIM = 128
HIDDEN_DIM = 128

token_embedding = Embedding(
    num_embeddings=vocab.get_vocab_size("tokens"), embedding_dim=EMBEDDING_DIM
)

# with open("saved_models/word2vec.th", "rb") as f:
#     token_embedding.load_state_dict(torch.load(f))

# token_embedding.weight.requires_grad = False

word_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

contextualizer = PytorchSeq2SeqWrapper(
    torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=False)
)
# contextualizer = IntraSentenceAttentionEncoder(
#     EMBEDDING_DIM,
#     num_attention_heads=4,
#     similarity_function=MultiHeadedSimilarity(4, EMBEDDING_DIM),
# )
# contextualizer = StackedSelfAttentionEncoder(
#     EMBEDDING_DIM,
#     hidden_dim=HIDDEN_DIM,
#     projection_dim=HIDDEN_DIM,
#     feedforward_hidden_dim=HIDDEN_DIM,
#     num_layers=1,
#     num_attention_heads=4,
# )

hparams = {"lr": 0.005, "batch_size": 32, "num_epochs": 20}
pred_metrics, model_saved_path = train(
    train_dataset, val_dataset, test_dataset, vocab, word_embedder, contextualizer, hparams
)

print(pred_metrics)

with open("saved_models/embeddings.th", "wb") as f:
    torch.save(token_embedding.state_dict(), f)

