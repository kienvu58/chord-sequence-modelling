from audio.generate import generate_score_and_audio, generate_bass_notes
from misc import *
import itertools
import pandas as pd
from preprocess_data import convert_to_chord_name
from prepare_dataset import (dataframe_to_note_set_progression,
                             split_data_by_phrase, get_dataset,
                             split_data_by_movement,
                             get_movement_dataset,
                             dataframe_to_root_progression,
                             dataframe_to_note_set_progression_no_inversion_no_ninth,
                             dataframe_to_chord_name_progression,
                             transpose_phrase_to_c_maj_or_a_min,
                             dataframe_to_note_set_progression_sorted)
from models.ngram_model import NgramModel
from models.lstm_model import LSTMModel
from models.general_lstm_model import GeneralLSTMModel
from evaluate import evaluate, accuracy
from vocab import Vocab
import logging
from voiceleading_utilities import nonbijective_vl
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

SHORT_PHRASE_LEN = 1
SKIP_REPETITIONS = True
SKIP_DOUBLE_REPETITIONS = True


def save_movement_datasets(process_data_func=dataframe_to_note_set_progression):
    train_phrases, val_phrases, test_phrases = split_data_by_movement(
        "data/phrases.txt")

    train_dataset = get_movement_dataset(
        "data/all_annotations.csv", train_phrases, process_data_func, augment=True,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS, skip_double_repetitions=SKIP_DOUBLE_REPETITIONS)
    val_dataset = get_movement_dataset(
        "data/all_annotations.csv", val_phrases, process_data_func, augment=False,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS, skip_double_repetitions=SKIP_DOUBLE_REPETITIONS)
    test_dataset = get_movement_dataset(
        "data/all_annotations.csv", test_phrases, process_data_func, augment=False,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS, skip_double_repetitions=SKIP_DOUBLE_REPETITIONS)

    with open("data/train_movements.txt", "w") as f:
        f.write("\n".join(train_dataset))
    with open("data/val_movements.txt", "w") as f:
        f.write("\n".join(val_dataset))
    with open("data/test_movements.txt", "w") as f:
        f.write("\n".join(test_dataset))


def save_phrase_datasets(process_data_func=dataframe_to_note_set_progression, prefix=""):
    train_phrases, val_phrases, test_phrases = split_data_by_phrase(
        "data/phrases.txt")

    train_dataset = get_dataset(
        "data/all_annotations.csv", train_phrases, process_data_func, augment=True,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS, skip_double_repetitions=SKIP_DOUBLE_REPETITIONS)
    val_dataset = get_dataset(
        "data/all_annotations.csv", val_phrases, process_data_func, augment=False,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS, skip_double_repetitions=SKIP_DOUBLE_REPETITIONS)
    test_dataset = get_dataset(
        "data/all_annotations.csv", test_phrases, process_data_func, augment=False,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS, skip_double_repetitions=SKIP_DOUBLE_REPETITIONS)

    with open("data/train_{}phrases.txt".format(prefix), "w") as f:
        f.write("\n".join(train_dataset))
    with open("data/val_{}phrases.txt".format(prefix), "w") as f:
        f.write("\n".join(val_dataset))
    with open("data/test_{}phrases.txt".format(prefix), "w") as f:
        f.write("\n".join(test_dataset))

def save_transposed_datasets():
    train_phrases, val_phrases, test_phrases = split_data_by_phrase(
        "data/phrases.txt")

    process_data_func = dataframe_to_note_set_progression
    augment_func = transpose_phrase_to_c_maj_or_a_min

    train_dataset = get_dataset(
        "data/all_annotations.csv", train_phrases, process_data_func, augment=True,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS, skip_double_repetitions=SKIP_DOUBLE_REPETITIONS, augment_func=augment_func)
    val_dataset = get_dataset(
        "data/all_annotations.csv", val_phrases, process_data_func, augment=True,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS, skip_double_repetitions=SKIP_DOUBLE_REPETITIONS, augment_func=augment_func)
    test_dataset = get_dataset(
        "data/all_annotations.csv", test_phrases, process_data_func, augment=True,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS, skip_double_repetitions=SKIP_DOUBLE_REPETITIONS, augment_func=augment_func)

    with open("data/train_transposed_phrases.txt", "w") as f:
        f.write("\n".join(train_dataset))
    with open("data/val_transposed_phrases.txt", "w") as f:
        f.write("\n".join(val_dataset))
    with open("data/test_transposed_phrases.txt", "w") as f:
        f.write("\n".join(test_dataset))



def convert_phrases_to_root_progression():
    phrase_list, _, _ = split_data_by_phrase(
        "data/phrases.txt", split_ratio=[1, 0, 0], shuffle=False)
    dataset = get_dataset(
        "data/all_annotations.csv", phrase_list, dataframe_to_root_progression,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS, skip_double_repetitions=SKIP_DOUBLE_REPETITIONS)

    with open("data/phrases_root.txt", "w") as f:
        f.write("\n".join(dataset))


def convert_phrases():
    phrase_list, _, _ = split_data_by_phrase(
        "data/phrases.txt", split_ratio=[1, 0, 0], shuffle=False)
    dataset = get_dataset(
        "data/all_annotations.csv", phrase_list, dataframe_to_note_set_progression,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS, skip_double_repetitions=SKIP_DOUBLE_REPETITIONS)

    with open("data/phrases_note_set.txt", "w") as f:
        f.write("\n".join(dataset))

    progression_list = []
    for phrase in dataset:
        progression = phrase.split(" ")
        progression = convert_to_chord_name(progression)
        progression_list.append(progression)

    with open("data/phrases_name.txt", "w") as f:
        f.write("\n".join(progression_list))


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
    print("generated progression:", len(progression),
        progression)
        #   convert_to_chord_name(progression))
    print("generated progression perplexity:", evaluate(model, [progression]))

    # print("accuracy:", accuracy(model, test))
    # generate_score_and_audio(progression, "ngram", "output")


def general_lstm_model(train, val, test):
    # vocab = Vocab()
    # vocab.from_dataset(train+val+test)
    vocab = generate_vocab()

    hparams = {
        "vocab": vocab,
        "pretrained_embedding": None,
        "freeze_embedding": True,
        "embedding_dim": 12,
        "hidden_dim": 128,
        "n_epochs": 5,
        "batch_size": 32,
        "lr": 0.001,
        "num_layers": 1,
        "dropout": 0,
    }

    model = GeneralLSTMModel(hparams)
    model.fit(train, val)
    print("lstm model perplexity:", evaluate(model, test))
    progression = model.generate(10)
    print("generated progression:", len(progression),
        progression)
        #   convert_to_chord_name(progression))
    print("generated progression perplexity:", evaluate(model, [progression]))
    print("accuracy:", accuracy(model, test))
    print("vocab size:", len(vocab))
    # generate_score_and_audio(progression, "lstm", "output")

def lstm_model(train, val, test):
    # vocab = Vocab()
    # vocab.from_dataset(train+val+test)
    vocab = generate_vocab()

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
    print("generated progression:", len(progression),
        progression)
        #   convert_to_chord_name(progression))
    print("generated progression perplexity:", evaluate(model, [progression]))
    print("accuracy:", accuracy(model, test))
    print("vocab size:", len(vocab))
    # generate_score_and_audio(progression, "lstm", "output")


def load_train_val_test(func, level="phrase"):
    train = func("data/train_{}s.txt".format(level))
    val = func("data/val_{}s.txt".format(level))
    test = func("data/test_{}s.txt".format(level))
    return train, val, test


def calculate_nonbijective_vl():
    with open("data/phrases_note_set.txt") as f:
        phrases = f.read().split("\n")

    vl_list = []
    for phrase in phrases:
        note_set_list = phrase.split(" ")
        vls = []
        for i in range(len(note_set_list) - 1):
            first_pcs = [int(note) for note in note_set_list[i].split("_")]
            second_pcs = [int(note) for note in note_set_list[i+1].split("_")]
            size, _ = nonbijective_vl(first_pcs, second_pcs)
            vls.append(str(size))
        vl_list.append(" ".join(vls))

    with open("data/phrases_vl.txt", "w") as f:
        f.write("\n".join(vl_list))


# save_phrase_datasets(process_data_func=dataframe_to_note_set_progression)
# save_movement_datasets(process_data_func=dataframe_to_note_set_progression)
# convert_phrases()
# convert_phrases_to_root_progression()
# generate_audio()
# calculate_nonbijective_vl()
# save_transposed_datasets()
# save_phrase_datasets(process_data_func=dataframe_to_note_set_progression_no_inversion_no_ninth, prefix="no_inversion_")
# train, val, test = load_train_val_test(load_dataset, level="no_inversion_phrase")
# save_phrase_datasets(process_data_func=dataframe_to_chord_name_progression, prefix="chord_name_")
train, val, test = load_train_val_test(load_dataset, level="chord_name_phrase")
ngram_model(train, val, test)
general_lstm_model(train, val, test)
