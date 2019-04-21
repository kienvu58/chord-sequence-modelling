import itertools
import pandas as pd
import random
import numpy as np
from preprocess_data import get_root, get_key_number, get_note_set, get_chord_name


def split_data_by_movement(phrase_txt, split_ratio=[7, 1, 2], skip_short_phrases=None):
    with open(phrase_txt) as f:
        movement_list = f.read().split("\n")

    movement_phrase_list = []

    for movement in movement_list:
        phrase_str_list = movement.split(" ")
        phrase_list = []
        for phrase_str in phrase_str_list[1:]:
            begin_idx, end_idx = phrase_str.split(":")
            begin_idx = int(begin_idx)
            end_idx = int(end_idx)

            phrase_len = end_idx - begin_idx
            if skip_short_phrases is None or phrase_len > skip_short_phrases:
                phrase_list.append((begin_idx, end_idx))
        movement_phrase_list.append(phrase_list)

    phrase_list = movement_phrase_list
    n_phrases = len(phrase_list)
    n_train = int(n_phrases * split_ratio[0]/sum(split_ratio))
    n_val = int(n_phrases * split_ratio[1]/sum(split_ratio))

    random.shuffle(phrase_list)

    train_phrases = phrase_list[:n_train]
    val_phrases = phrase_list[n_train:n_train+n_val]
    test_phrases = phrase_list[n_train+n_val:]
    return train_phrases, val_phrases, test_phrases


def split_data_by_phrase(phrases_txt, split_ratio=[7, 1, 2], shuffle=True):
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
            phrase_list.append((begin_idx, end_idx))

    n_phrases = len(phrase_list)
    n_train = int(n_phrases * split_ratio[0]/sum(split_ratio))
    n_val = int(n_phrases * split_ratio[1]/sum(split_ratio))

    if shuffle:
        random.shuffle(phrase_list)

    train_phrases = phrase_list[:n_train]
    val_phrases = phrase_list[n_train:n_train+n_val]
    test_phrases = phrase_list[n_train+n_val:]
    return train_phrases, val_phrases, test_phrases


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


def transpose_phrase_to_c_maj_or_a_min(df):
    assert len(df["global_key"].unique()) == 1
    original_key = df["global_key"].iloc[0]
    is_minor_key = (original_key == original_key.lower())
    key = "a" if is_minor_key else "C"
    transposed_df = df.replace(
        to_replace={"global_key": original_key}, value=key, inplace=False)

    return [transposed_df]


def get_movement_dataset(all_csv, movement_phrase_list, process_data_func, augment=False, skip_short_phrases=0, skip_repetitions=False):
    df_all = pd.read_csv(all_csv)
    dataset = []

    for phrase_list in movement_phrase_list:
        frames = []
        for beg, end in phrase_list:
            df = df_all[beg:end]
            if end - beg > skip_short_phrases:
                frames.append(df)
        df_movement = pd.concat(frames)

        if augment:
            df_list = transpose_phrase(df_movement)
        else:
            df_list = [df_movement]

        for df in df_list:
            progression = process_data_func(df)
            dataset.append(progression)

    final_dataset = []
    for progression in dataset:
        chord_list = progression.split(" ")
        if skip_repetitions:
            chord_list = [k for k, g in itertools.groupby(chord_list)]
        if len(chord_list) > skip_short_phrases:
            final_dataset.append(" ".join(chord_list))

    return final_dataset


def get_dataset(all_csv, phrase_list, process_data_func, augment=False,
                skip_short_phrases=0, skip_repetitions=False, skip_double_repetitions=False, augment_func=transpose_phrase):
    """
    process_data_func: take a dataframe as input and return a chord progression
        e.g. process_data_func = lambda df: some_func(df, *args)
    """
    df_all = pd.read_csv(all_csv)
    dataset = []

    for beg, end in phrase_list:
        df = df_all[beg:end]
        if augment:
            df_list = augment_func(df)
        else:
            df_list = [df]

        for df in df_list:
            progression = process_data_func(df)
            dataset.append(progression)

    final_dataset = []
    for progression in dataset:
        chord_list = progression.split(" ")
        if skip_repetitions:
            chord_list = [k for k, g in itertools.groupby(chord_list)]

        if skip_double_repetitions:
            n = len(chord_list)
            i = 0
            tmp_chord_list = []
            while i < n:
                if 1 < i and i+1 < n:
                    if chord_list[i-2] == chord_list[i] and chord_list[i-1] == chord_list[i+1]:
                        i += 2
                        continue
                    else:
                        tmp_chord_list.append(chord_list[i])
                        i += 1
                else:
                    tmp_chord_list.append(chord_list[i])
                    i += 1
            chord_list = tmp_chord_list

        if len(chord_list) > skip_short_phrases:
            final_dataset.append(" ".join(chord_list))
    return final_dataset


def join_list(note_set):
    return "_".join([str(n) for n in note_set])


def chord_name(row):
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

    name = get_chord_name(global_key, local_key, numeral,
                          form, figbass, relativeroot, True)
    return name


def convert_to_figured_bass(row):
    note_set = convert_to_note_set(row)
    if not isinstance(note_set, str):
        return np.nan
    bass_note_index = int(note_set.split("_")[0])
    major_key_list = ["C", "Db", "D", "Eb", "E",
                      "F", "F#", "G", "Ab", "A", "Bb", "B"]
    return major_key_list[bass_note_index]


def convert_to_note_set(row, add_root=True, sort_notes=False, root_only=False, use_inversion=True, use_ninth=True):
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
                            figbass, changes, use_changes=False, use_inversion=use_inversion, use_ninth=use_ninth)

    if root_only:
        major_key_list = ["C", "Db", "D", "Eb", "E",
                          "F", "F#", "G", "Ab", "A", "Bb", "B"]
        minor_key_list = ["c", "c#", "d", "eb", "e",
                          "f", "f#", "g", "g#", "a", "bb", "b"]
        if quality == "minor":
            return minor_key_list[root_number]
        elif quality == "major":
            return major_key_list[root_number]
        local_key_minor = local_key == local_key.lower()
        if local_key_minor:
            return minor_key_list[root_number]
        else:
            return major_key_list[root_number]

    if add_root:
        note_set = [(n + root_number) % 12 for n in note_set]
    else:
        note_set = [root_number] + note_set

    if sort_notes:
        note_set.sort()

    return join_list(note_set)


def dataframe_to_note_set_progression_sorted(df):
    note_set_progression = df.apply(
        lambda row: convert_to_note_set(row, sort_notes=True), axis=1)
    return " ".join([note for note in note_set_progression if not pd.isnull(note)])


def dataframe_to_note_set_progression_no_inversion_no_ninth(df):
    note_set_progression = df.apply(
        lambda row: convert_to_note_set(row, use_inversion=False, use_ninth=False), axis=1)
    return " ".join([note for note in note_set_progression if not pd.isnull(note)])


def dataframe_to_note_set_progression(df):
    note_set_progression = df.apply(
        lambda row: convert_to_note_set(row), axis=1)
    return " ".join([note for note in note_set_progression if not pd.isnull(note)])


def dataframe_to_root_progression(df):
    root_progression = df.apply(
        lambda row: convert_to_note_set(row, root_only=True),
        axis=1
    )
    return " ".join([note for note in root_progression if not pd.isnull(note)])


def dataframe_to_figured_bass_progression(df):
    progression = df.apply(convert_to_figured_bass, axis=1)
    return " ".join([note for note in progression if not pd.isnull(note)])


def dataframe_to_chord_name_progression(df):
    progression = df.apply(chord_name, axis=1)
    return " ".join([chord for chord in progression if not pd.isnull(chord)])
