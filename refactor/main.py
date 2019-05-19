import torch
import torch.optim as optim
import numpy as np
import shutil
import itertools
import json
import time
import math

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

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

torch.manual_seed(1)

target_weight = "data/transformer_weight.th"


def run_experiment(
    use_soft_targets, soft_target_path, embedding_type, rnn_type, hparams
):
    log = {}
    log["name"] = "{} {} {}".format(
        rnn_type, embedding_type, "soft_target" if use_soft_targets else "hard_target"
    )
    log["soft_target"] = soft_target_path if use_soft_targets else None

    vocab = Vocabulary().from_files(hparams["vocab_path"])
    if embedding_type == "Chord":
        # data reader
        reader = CpmDatasetReader()

        # chord embedder
        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size("tokens"),
            embedding_dim=hparams["chord_token_embedding_dim"],
        )
        chord_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

    elif embedding_type == "Note":
        # data reader
        note_tokenizer = NoteTokenizer()
        note_indexer = TokenCharactersIndexer(
            namespace="notes", min_padding_length=4, character_tokenizer=note_tokenizer
        )
        reader = CpmDatasetReader(
            token_indexers={"tokens": SingleIdTokenIndexer(), "notes": note_indexer}
        )

        # chord embedder
        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size("tokens"),
            embedding_dim=hparams["chord_token_embedding_dim"],
        )
        note_token_embedding = Embedding(
            vocab.get_vocab_size("notes"), hparams["note_embedding_dim"]
        )
        note_encoder = CnnEncoder(
            num_filters=hparams["cnn_encoder_num_filters"],
            ngram_filter_sizes=hparams["cnn_encoder_n_gram_filter_sizes"],
            embedding_dim=hparams["note_embedding_dim"],
            output_dim=hparams["note_level_embedding_dim"],
        )
        note_embedding = TokenCharactersEncoder(note_token_embedding, note_encoder)
        chord_embedder = BasicTextFieldEmbedder(
            {"tokens": token_embedding, "notes": note_embedding}
        )
    else:
        raise ValueError("Unknown embedding type:", embedding_type)

    # read data
    train_dataset = reader.read(os.path.join(hparams["data_path"], "train.txt"))
    val_dataset = reader.read(os.path.join(hparams["data_path"], "val.txt"))
    test_dataset = reader.read(os.path.join(hparams["data_path"], "test.txt"))

    # contextualizer
    contextual_input_dim = chord_embedder.get_output_dim()
    if rnn_type == "RNN":
        contextualizer = PytorchSeq2SeqWrapper(
            torch.nn.RNN(
                contextual_input_dim,
                hparams["rnn_hidden_dim"],
                batch_first=True,
                bidirectional=False,
            )
        )
    elif rnn_type == "LSTM":
        contextualizer = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(
                contextual_input_dim,
                hparams["lstm_hidden_dim"],
                batch_first=True,
                bidirectional=False,
            )
        )
    elif rnn_type == "GRU":
        contextualizer = PytorchSeq2SeqWrapper(
            torch.nn.GRU(
                contextual_input_dim,
                hparams["gru_hidden_dim"],
                batch_first=True,
                bidirectional=False,
            )
        )
    else:
        raise ValueError("Unknown rnn type:", rnn_type)

    if use_soft_targets:
        vocab_size = vocab.get_vocab_size("tokens")
        soft_targets = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=vocab_size,
            weight=torch.load(soft_target_path),
            trainable=False,
        )
    else:
        soft_targets = None

    iterator = BucketIterator(
        batch_size=hparams["batch_size"], sorting_keys=[("input_tokens", "num_tokens")]
    )
    iterator.index_with(vocab)

    batches_per_epoch = math.ceil(len(train_dataset) / hparams["batch_size"])

    model_hparams = {
        "dropout": None,
        "soft_targets": soft_targets,
        "T_initial": hparams["T_initial"],
        "decay_rate": hparams["decay_rate"],
        "batches_per_epoch": batches_per_epoch,
    }
    # chord progression model
    model = Cpm(vocab, chord_embedder, contextualizer, model_hparams)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
        print("GPU available.")
    else:
        cuda_device = -1

    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])

    ts = time.gmtime()
    saved_model_path = os.path.join(
        hparams["saved_model_path"], time.strftime("%Y-%m-%d %H-%M-%S", ts)
    )
    serialization_dir = os.path.join(saved_model_path, "checkpoints")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        serialization_dir=serialization_dir,
        patience=hparams["patience"],
        num_epochs=hparams["num_epochs"],
        cuda_device=cuda_device,
    )
    trainer.train()
    saved_model_path = os.path.join(saved_model_path, "{}.th".format(log["name"]))
    torch.save(model.state_dict(), saved_model_path)

    predictor = Predictor(model=model, iterator=iterator, cuda_device=cuda_device)
    pred_metrics = predictor.predict(test_dataset)
    log["metrics"] = pred_metrics
    log["saved_mode_path"] = saved_model_path

    return log


def main():
    data_path = "data/cv/0/"
    vocab_path = "data/vocabulary/"
    saved_model_path = "saved_models/"
    hparams = {
        "lr": 0.001,
        "batch_size": 8,
        "num_epochs": 200,
        "patience": 20,
        "rnn_hidden_dim": 128,
        "lstm_hidden_dim": 128,
        "gru_hidden_dim": 128,
        "chord_token_embedding_dim": 128,
        "note_embedding_dim": 64,
        "note_level_embedding_dim": 64,
        "cnn_encoder_num_filters": 16,
        "cnn_encoder_n_gram_filter_sizes": (2, 3, 4),
        "soft_target_path": soft_target_path,
        "T_initial": 5,
        "decay_rate": 0.01,
        "data_path": data_path,
        "vocab_path": vocab_path,
        "saved_model_path": saved_model_path,
    }

    embedding_type_list = ["Chord", "Note"]
    rnn_type_list = ["RNN", "LSTM", "GRU"]
    soft_target_path_list = ["data/targetes/target_533313.th"]
    use_soft_targets = False

    result = {}
    result["experiments"] = []
    for embedding_type, rnn_type, soft_target_path in itertools.product(
        embedding_type_list, rnn_type_list, soft_target_path_list
    ):
        log = run_experiment(
            use_soft_targets, soft_target_path, embedding_type, rnn_type, hparams
        )
        result["experiments"].append(log)
    result["hparams"] = hparams
    ts = time.gmtime()
    result_fn = "{}.json".format(time.strftime("%Y-%m-%d %H-%M-%S", ts))

    with open(os.path.join("logs", result_fn), "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
