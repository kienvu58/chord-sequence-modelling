from audio.generate import generate_score_and_audio, generate_bass_notes
import pandas as pd
from preprocess_data import convert_to_chord_name
from prepare_dataset import dataframe_to_note_set_progression, split_data_by_phrase, get_dataset, split_data_by_movement, get_movement_dataset
from models.ngram_model import NgramModel
from models.lstm_model import LSTMModel
from evaluate import evaluate
from vocab import Vocab
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def save_movement_datasets():
    train_phrases, val_phrases, test_phrases = split_data_by_movement(
        "data/phrases.txt", skip_short_phrases=1)

    def process_data_func(df): return dataframe_to_note_set_progression(df)

    train_dataset = get_movement_dataset(
        "data/all_annotations.csv", train_phrases, process_data_func, augment=True)
    val_dataset = get_movement_dataset(
        "data/all_annotations.csv", val_phrases, process_data_func, augment=False)
    test_dataset = get_movement_dataset(
        "data/all_annotations.csv", test_phrases, process_data_func, augment=False)

    with open("data/train_movements.txt", "w") as f:
        f.write("\n".join(train_dataset))
    with open("data/val_movements.txt", "w") as f:
        f.write("\n".join(val_dataset))
    with open("data/test_movements.txt", "w") as f:
        f.write("\n".join(test_dataset))


def convert_phrases():
    phrase_list, _, _ = split_data_by_phrase("data/phrases.txt", split_ratio=[1, 0, 0], shuffle=False)
    dataset = get_dataset("data/all_annotations.csv", phrase_list, dataframe_to_note_set_progression)

    progression_list = []
    for phrase in dataset:
        progression = phrase.split(" ")
        progression = convert_to_chord_name(progression)
        progression_list.append(progression)

    with open("data/phrases_name.txt", "w") as f:
        f.write("\n".join(progression_list))


def save_phrase_datasets():
    train_phrases, val_phrases, test_phrases = split_data_by_phrase(
        "data/phrases.txt", skip_short_phrases=1)

    def process_data_func(df): return dataframe_to_note_set_progression(df)
    train_dataset = get_dataset(
        "data/all_annotations.csv", train_phrases, process_data_func, augment=True)
    val_dataset = get_dataset("data/all_annotations.csv",
                              val_phrases, process_data_func, augment=False)
    test_dataset = get_dataset(
        "data/all_annotations.csv", test_phrases, process_data_func, augment=False)

    with open("data/train_phrases.txt", "w") as f:
        f.write("\n".join(train_dataset))
    with open("data/val_phrases.txt", "w") as f:
        f.write("\n".join(val_dataset))
    with open("data/test_phrases.txt", "w") as f:
        f.write("\n".join(test_dataset))


def load_dataset(filename):
    with open(filename) as f:
        dataset = f.read().split("\n")
    dataset = [p.split(" ") for p in dataset]
    return dataset


def load_base_note_dataset(filename):
    dataset = load_dataset(filename)
    dataset = [[note_set.split("_")[0] for note_set in p] for p in dataset]
    return dataset


def generate_audio():
    df_all = pd.read_csv("all_annotations.csv")
    first_movement = df_all.iloc[0:558]
    progression = dataframe_to_note_set_progression(first_movement)
    # generate_bass_notes(progression, "12_1", "output")
    generate_score_and_audio(progression, "12_1", "output")


def ngram_model(train, val, test):
    hparams = {
        "order": 2,
        "counter": None,
        "vocab": None,
        "model_class": "Laplace",
        "kwargs": {}
    }
    model = NgramModel(hparams)
    model.fit(train)
    print("ngram model perplexity:", evaluate(model, test))
    progression = model.generate(10)
    print("generated progression:", len(progression), convert_to_chord_name(progression))
    print("generated progression perplexity:", evaluate(model, [progression]))
    generate_score_and_audio(progression, "ngram", "output")


def lstm_model(train, val, test):
    vocab = Vocab()
    vocab.from_dataset(train+val+test)

    hparams = {
        "vocab": vocab,
        "embedding_dim": 128,
        "hidden_dim": 128,
        "n_epochs": 5,
        "batch_size": 32,
        "lr": 0.001,
        "num_layers": 1,
        "dropout": 0,
    }

    model = LSTMModel(hparams)
    model.fit(train, val)
    print("lstm model perplexity:", evaluate(model, test))
    progression = model.generate(10)
    print("generated progression:", len(progression), convert_to_chord_name(progression))
    print("generated progression perplexity:", evaluate(model, [progression]))
    generate_score_and_audio(progression, "lstm", "output")


def load_train_val_test(func, level="phrase"):
    train = func("data/train_{}s.txt".format(level))
    val = func("data/val_{}s.txt".format(level))
    test = func("data/test_{}s.txt".format(level))
    return train, val, test


convert_phrases()
# save_movement_datasets()
# generate_audio()
# train, val, test = load_train_val_test(load_dataset, level="phrase")
# ngram_model(train, val, test)
# lstm_model(train, val, test)
