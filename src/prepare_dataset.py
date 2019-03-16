import pandas as pd
import random
import numpy as np
from preprocess_data import get_root, get_key_number, get_note_set


def split_data_by_phrase(phrases_txt, split_ratio=0.7, skip_short_phrases=None):
    """
    returns 2 lists: train and test set
        each list contains tuples of begin and end indices of phrases
    """
    with open(phrases_txt) as f:
        movement_list = f.read().split("\n")

    phrase_list = []
    for movement in movement_list:
        phrase_str_list = movement.split(" ")
        for phrase_str in phrase_str_list[1:]:
            begin_idx, end_idx = phrase_str.split(":")
            begin_idx = int(begin_idx)
            end_idx = int(end_idx)

            phrase_len = end_idx - begin_idx
            if skip_short_phrases is None or phrase_len > skip_short_phrases:
                phrase_list.append((begin_idx, end_idx))

    n_phrases = len(phrase_list)
    n_train = int(n_phrases * split_ratio)

    random.shuffle(phrase_list)

    train_phrases = phrase_list[:n_train]
    test_phrases = phrase_list[n_train:]
    return train_phrases, test_phrases


def transpose_phrase(df):
    """
    returns a list of 12 dfs of 12 global keys
    """
    assert len(df["global_key"].unique()) == 1
    original_key = df["global_key"].iloc[0]
    is_minor_key = (original_key == original_key.lower())
    major_key_list = ["C", "G", "D", "A", "E",
                      "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
    minor_key_list = ["a", "e", "b", "f#", "c#",
                      "g#", "eb", "bb", "f", "c", "g", "d"]
    key_list = minor_key_list if is_minor_key else major_key_list
    transposed_list = []
    for key in key_list:
        transposed_df = df.replace(
            to_replace={"global_key": original_key}, value=key, inplace=False)
        transposed_list.append(transposed_df)

    return transposed_list


def get_dataset(all_csv, phrase_list, process_data_func, augment=False):
    """
    process_data_func: take a dataframe as input and return a chord progression
        e.g. process_data_func = lambda df: some_func(df, *args)
    """
    df_all = pd.read_csv(all_csv)
    dataset = []

    for beg, end in phrase_list:
        df = df_all[beg:end]
        if augment:
            df_list = transpose_phrase(df)
        else:
            df_list = [df]

        for df in df_list:
            progression = process_data_func(df)
            dataset.append(progression)
    return dataset


def join_list(note_set):
    return "_".join([str(n) for n in note_set])


def convert_to_note_set(row, add_root=True, sort_notes=False):
    if pd.isnull(row["numeral"]):
        return np.nan

    global_key = row["global_key"]
    local_key = row["local_key"]
    numeral = row["numeral"]
    form = row["form"] if not pd.isnull(row["form"]) else None
    figbass = str(int(row["figbass"])) if not pd.isnull(
        row["figbass"]) else None
    relativeroot = row["relativeroot"] if not pd.isnull(
        row["relativeroot"]) else None
    changes = str(row["changes"]) if not pd.isnull(row["changes"]) else None

    if numeral in ["Ger", "It", "Fr"]:
        quality = numeral
    else:
        quality = "minor" if numeral == numeral.lower() else "major"

    major = not (local_key == local_key.lower())
    root = get_root(global_key, local_key, numeral, relativeroot)
    root_number = get_key_number(root)
    note_set = get_note_set(quality, major, form,
                            figbass, changes, use_changes=False)

    if add_root:
        note_set = [(n + root_number) % 12 for n in note_set]
    else:
        note_set = [root_number] + note_set

    if sort_notes:
        note_set.sort()

    return join_list(note_set)


def dataframe_to_note_set_progression(df):
    note_set_progression = df.apply(
        lambda row: convert_to_note_set(row), axis=1)
    return " ".join([note for note in note_set_progression if not pd.isnull(note)])