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

def examine_note_embedding_models(model_saved_path):
    token_embedding_dim = 128
    lstm_hidden_dim = 128

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
        num_filters=16, ngram_filter_sizes=(2, 3, 4), embedding_dim=64, output_dim=64
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

    model = Cpm(vocab, word_embedder, contextualizer)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    print(cuda_device)

    iterator = BucketIterator(batch_size=32, sorting_keys=[("input_tokens", "num_tokens")])
    iterator.index_with(vocab)

    model.load_state_dict(torch.load(model_saved_path))

    predictor = Predictor(model=model, iterator=iterator, cuda_device=cuda_device)
    pred_metrics = predictor.predict(train_dataset)
    predictor.save_confusion_matrices()

def examine_note_embedding_models_with_soft_targets(model_saved_path):
    token_embedding_dim = 128
    lstm_hidden_dim = 128

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
        num_filters=16, ngram_filter_sizes=(2, 3, 4), embedding_dim=64, output_dim=64
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

    vocab_size = vocab.get_vocab_size("tokens")
    soft_targets = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=vocab_size,
        weight=torch.load("data/transformer_weight.th"),
        trainable=False
    )

    model = Cpm(vocab, word_embedder, contextualizer, soft_targets=soft_targets)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    print(cuda_device)

    iterator = BucketIterator(batch_size=32, sorting_keys=[("input_tokens", "num_tokens")])
    iterator.index_with(vocab)

    model.load_state_dict(torch.load(model_saved_path))

    predictor = Predictor(model=model, iterator=iterator, cuda_device=cuda_device)
    pred_metrics = predictor.predict(train_dataset)
    predictor.save_confusion_matrices()

def examine_baseline_models_with_soft_targets(model_saved_path):
    token_embedding_dim = 128
    lstm_hidden_dim = 128

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
    soft_targets = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=vocab_size,
        weight=torch.load("data/transformer_weight.th"),
        trainable=False
    )

    model = Cpm(vocab, word_embedder, contextualizer, soft_targets=soft_targets)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    print(cuda_device)

    iterator = BucketIterator(batch_size=32, sorting_keys=[("input_tokens", "num_tokens")])
    iterator.index_with(vocab)

    model.load_state_dict(torch.load(model_saved_path))

    predictor = Predictor(model=model, iterator=iterator, cuda_device=cuda_device)
    pred_metrics = predictor.predict(test_dataset)
    predictor.save_confusion_matrices()

# examine_note_embedding_models("saved_models/note_embedding_lstm_192_128.th")
# examine_baseline_models_with_soft_targets("saved_models/baseline_lstm_128_128.th")
examine_note_embedding_models_with_soft_targets("saved_models/note_embedding_lstm_192_128_with_soft_targets.th")