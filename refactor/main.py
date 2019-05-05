import torch
import torch.optim as optim
import numpy as np
import shutil

from modules.tokenizers import ChordCharacterTokenizer, NoteTokenizer
from modules.dataset_readers import CpmDatasetReader
from modules.chord_progression_models import Cpm
from modules.predictors import Predictor

from allennlp.training.learning_rate_schedulers import CosineWithRestarts
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import (
    TextFieldEmbedder,
    BasicTextFieldEmbedder,
)
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import (
    AugmentedLstm,
    BagOfEmbeddingsEncoder,
    CnnEncoder,
    CnnHighwayEncoder,
    PytorchSeq2VecWrapper,
)
from allennlp.modules.seq2seq_encoders import (
    Seq2SeqEncoder,
    PytorchSeq2SeqWrapper,
    IntraSentenceAttentionEncoder,
    StackedSelfAttentionEncoder,
)
from allennlp.modules.similarity_functions import MultiHeadedSimilarity
from allennlp.common.file_utils import cached_path
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import (
    TokenIndexer,
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
)
from allennlp.training.trainer import Trainer
import logging
import sys
import os

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
    target_transformer=None,
    model_saved_path="saved_models/tmp.th",
):
    model = Cpm(
        vocab,
        word_embedder,
        contextualizer,
        dropout=contextual_embedding_dropout,
        target_transformer=target_transformer,
    )
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

    serialization_dir = "saved_models/checkpoints"
    if os.path.isdir(serialization_dir):
        shutil.rmtree(serialization_dir)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        serialization_dir=serialization_dir,
        patience=10,
        num_epochs=hparams["num_epochs"],
        cuda_device=cuda_device,
    )
    trainer.train()
    torch.save(model.state_dict(), model_saved_path)

    predictor = Predictor(model=model, iterator=iterator, cuda_device=cuda_device)
    pred_metrics = predictor.predict(test_dataset)
    return pred_metrics


def baseline_lstm(hparams, token_embedding_dim=128, lstm_hidden_dim=128):
    reader = CpmDatasetReader()
    train_dataset = reader.read("data/cv/0/train.txt")
    val_dataset = reader.read("data/cv/0/val.txt")
    test_dataset = reader.read("data/cv/0/test.txt")

    vocab = Vocabulary().from_files("data/vocabulary")
    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size("tokens"), embedding_dim=token_embedding_dim
    )
    word_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
    contextual_input_dim = word_embedder.get_output_dim()
    contextualizer = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(
            contextual_input_dim, lstm_hidden_dim, batch_first=True, bidirectional=False
        )
    )

    pred_metrics = train(
        train_dataset,
        val_dataset,
        test_dataset,
        vocab,
        word_embedder,
        contextualizer,
        hparams,
        model_saved_path="saved_models/baseline_lstm_{}_{}.th".format(
            contextual_input_dim, lstm_hidden_dim
        ),
    )
    print(pred_metrics)


def baseline_lstm_with_target_transformer(
    hparams, token_embedding_dim=128, lstm_hidden_dim=128
):
    reader = CpmDatasetReader()
    train_dataset = reader.read("data/cv/0/train.txt")
    val_dataset = reader.read("data/cv/0/val.txt")
    test_dataset = reader.read("data/cv/0/test.txt")

    vocab = Vocabulary().from_files("data/vocabulary")
    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size("tokens"), embedding_dim=token_embedding_dim
    )
    word_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
    contextual_input_dim = word_embedder.get_output_dim()
    contextualizer = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(
            contextual_input_dim, lstm_hidden_dim, batch_first=True, bidirectional=False
        )
    )

    vocab_size = vocab.get_vocab_size("tokens")
    target_transformer = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=vocab_size,
        weight=torch.load("data/transformer_weight.th"),
        trainable=False
    )

    pred_metrics = train(
        train_dataset,
        val_dataset,
        test_dataset,
        vocab,
        word_embedder,
        contextualizer,
        hparams,
        target_transformer=target_transformer,
        model_saved_path="saved_models/baseline_lstm_{}_{}.th".format(
            contextual_input_dim, lstm_hidden_dim
        ),
    )
    print(pred_metrics)


def character_embedding_lstm(hparams, token_embedding_dim=128, lstm_hidden_dim=128):
    chord_character_tokenizer = ChordCharacterTokenizer()
    token_characters_indexer = TokenCharactersIndexer(
        min_padding_length=3, character_tokenizer=chord_character_tokenizer
    )
    reader = CpmDatasetReader(
        token_indexers={
            "tokens": SingleIdTokenIndexer(),
            "token_characters": token_characters_indexer,
        }
    )
    train_dataset = reader.read("data/cv/0/train.txt")
    val_dataset = reader.read("data/cv/0/val.txt")
    test_dataset = reader.read("data/cv/0/test.txt")

    vocab = Vocabulary().from_files("data/vocabulary")

    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size("tokens"), embedding_dim=token_embedding_dim
    )
    character_tokens_embedding = Embedding(vocab.get_vocab_size("token_characters"), 64)
    chracters_encoder = CnnEncoder(
        num_filters=16, ngram_filter_sizes=(2, 3, 4), embedding_dim=64, output_dim=64
    )
    characters_embedding = TokenCharactersEncoder(
        character_tokens_embedding, chracters_encoder
    )
    word_embedder = BasicTextFieldEmbedder(
        {"tokens": token_embedding, "token_characters": characters_embedding}
    )

    contextual_input_dim = word_embedder.get_output_dim()
    contextualizer = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(
            contextual_input_dim, lstm_hidden_dim, batch_first=True, bidirectional=False
        )
    )

    pred_metrics = train(
        train_dataset,
        val_dataset,
        test_dataset,
        vocab,
        word_embedder,
        contextualizer,
        hparams,
        model_saved_path="saved_models/character_embedding_lstm_{}_{}.th".format(
            contextual_input_dim, lstm_hidden_dim
        ),
    )
    print(pred_metrics)


def note_embedding_lstm(hparams, token_embedding_dim=128, lstm_hidden_dim=128):
    note_tokenizer = NoteTokenizer()
    note_indexer = TokenCharactersIndexer(
        namespace="notes", min_padding_length=4, character_tokenizer=note_tokenizer
    )
    reader = CpmDatasetReader(
        token_indexers={"tokens": SingleIdTokenIndexer(), "notes": note_indexer}
    )
    train_dataset = reader.read("data/cv/0/train.txt")
    val_dataset = reader.read("data/cv/0/val.txt")
    test_dataset = reader.read("data/cv/0/test.txt")

    vocab = Vocabulary().from_files("data/vocabulary")

    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size("tokens"), embedding_dim=token_embedding_dim
    )
    note_token_embedding = Embedding(vocab.get_vocab_size("notes"), 64)
    note_encoder = CnnEncoder(
        num_filters=16, ngram_filter_sizes=(2, 3), embedding_dim=64, output_dim=64
    )
    note_embedding = TokenCharactersEncoder(note_token_embedding, note_encoder)
    word_embedder = BasicTextFieldEmbedder(
        {"tokens": token_embedding, "notes": note_embedding}
    )

    contextual_input_dim = word_embedder.get_output_dim()
    contextualizer = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(
            contextual_input_dim, lstm_hidden_dim, batch_first=True, bidirectional=False
        )
    )

    pred_metrics = train(
        train_dataset,
        val_dataset,
        test_dataset,
        vocab,
        word_embedder,
        contextualizer,
        hparams,
        model_saved_path="saved_models/note_embedding_lstm_{}_{}.th".format(
            contextual_input_dim, lstm_hidden_dim
        ),
    )
    print(pred_metrics)


hparams = {"lr": 0.001, "batch_size": 8, "num_epochs": 500}
# baseline_lstm(hparams)
# character_embedding_lstm(hparams)
# note_embedding_lstm(hparams)
baseline_lstm_with_target_transformer(hparams)

