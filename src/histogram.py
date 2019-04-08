
# convert dataset into phrases
#   no augmented
#   transpose to C major or C minor key

# create a table of transition
#   count the number of each transition
import itertools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from preprocess_data import convert_to_chord_name
from prepare_dataset import (dataframe_to_note_set_progression,
                             split_data_by_phrase, get_dataset,
                             split_data_by_movement,
                             get_movement_dataset,
                             dataframe_to_note_set_progression_sorted,
                             dataframe_to_root_progression,
                             transpose_phrase_to_c_maj_or_a_min)

SHORT_PHRASE_LEN = 1
SKIP_REPETITIONS = True
TRANSPOSE = True
if TRANSPOSE:
    SUFFIX = "_after_transposing"
else:
    SUFFIX = ""


def histogram_numerals():
    df = pd.read_csv("data/all_annotations.csv")
    n_chords = len(df)
    table = np.zeros((7, 9))
    lut = {
        0: ["I", "i"],
        1: ["II", "ii"],
        2: ["III", "iii"],
        3: ["IV", "iv"],
        4: ["V", "v"],
        5: ["VI", "vi"],
        6: ["VII", "vii"],
    }

    for i in range(n_chords-1):
        will_modulate = df.loc[i]["local_key"] != df.loc[i+1]["local_key"]
        is_relative_chord = not pd.isnull(df.loc[i]["relativeroot"])
        next_is_relative_chord = not pd.isnull(df.loc[i+1]["relativeroot"])
        if is_relative_chord or next_is_relative_chord:
            continue

        start = -1
        for k, v in lut.items():
            if df.loc[i]["numeral"] in v:
                start = k
                break
        if start == -1:
            continue

        if will_modulate:
            table[start][8] += 1

        if df.loc[i]["phraseend"]:
            table[start][7] += 1
            continue

        end = -1
        for k, v in lut.items():
            if df.loc[i+1]["numeral"] in v:
                end = k
                break
        if end == -1:
            continue

        table[start][end] += 1

    start = -1
    for k, v in lut.items():
        if df.loc[n_chords-1]["numeral"] in v:
            start = k
            break
    if start != -1:
        table[start][7] += 1

    return table, n_chords


def print_result_histogram_numerals(table, n_chords):
    for i in range(7):
        table[i][i] = 0.0

    n_transitions = np.sum(table[:-1, :])
    columns = ["I", "II", "III", "IV", "V", "VI", "VII", "END", "MOD"]
    rows = ["I", "II", "III", "IV", "V", "VI", "VII"]

    print("Transition ratio:", n_transitions/(n_chords-1))
    row_sum = np.sum(table, axis=1)
    table = table / row_sum[:, None]
    df = pd.DataFrame(table)
    df["sum"] = pd.Series(row_sum)
    print(df)

    plt.imshow(table, cmap="jet")
    plt.xticks(np.arange(len(columns)), columns)
    plt.yticks(np.arange(len(rows)), rows)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            plt.text(j, i, "{:.1f}".format(
                table[i][j]*100), ha="center", va="center", color="w")

    plt.savefig(
        "figures/diatonic_scale_transitions ({:.1f}%).pdf".format(n_transitions/(n_chords-1)*100))


def histogram_chords(chord_output, is_sorted=True):
    if is_sorted:
        process_data_func = dataframe_to_note_set_progression_sorted
    else:
        process_data_func = dataframe_to_note_set_progression

    phrase_list, _, _ = split_data_by_phrase(
        "data/phrases.txt", split_ratio=[1, 0, 0], shuffle=False)
    dataset = get_dataset(
        "data/all_annotations.csv", phrase_list, process_data_func,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS, augment=True, augment_func=transpose_phrase_to_c_maj_or_a_min)

    unique_chords = set()
    progression_list = []
    for phrase in dataset:
        progression = phrase.split(" ")
        progression = convert_to_chord_name(
            progression, output=chord_output, is_sorted=is_sorted)
        progression = progression.split(" ")
        unique_chords.update(progression)
        progression_list.append(progression)

    unique_chords = list(unique_chords)
    unique_chords = sorted(unique_chords)
    n_rows = len(unique_chords)
    n_cols = n_rows + 1
    table = np.zeros((n_rows, n_cols))

    for progression in progression_list:
        for i in range(len(progression) - 1):
            start_chord = progression[i]
            end_chord = progression[i+1]
            start = unique_chords.index(start_chord)
            end = unique_chords.index(end_chord)
            table[start][end] += 1
        last = unique_chords.index(progression[-1])
        table[last][n_cols-1] += 1

    return table, unique_chords


def print_histogram_root(name="root_histogram"):
    if TRANSPOSE:
        augment_func = transpose_phrase_to_c_maj_or_a_min
    else:
        augment_func = None

    phrase_list, _, _ = split_data_by_phrase(
        "data/phrases.txt", split_ratio=[1, 0, 0], shuffle=False)
    dataset = get_dataset(
        "data/all_annotations.csv", phrase_list, dataframe_to_root_progression,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS, augment=TRANSPOSE, augment_func=augment_func)

    unique_chords = set()
    progression_list = []
    for phrase in dataset:
        progression = phrase.split(" ")
        unique_chords.update(progression)
        progression_list.append(progression)

    unique_chords = list(unique_chords)
    unique_chords = sorted(unique_chords)
    n_rows = len(unique_chords)
    counter = np.zeros((n_rows,))

    for progression in progression_list:
        for root in progression:
            index = unique_chords.index(root)
            counter[index] += 1

    plt.rcParams['figure.figsize'] = (20, 20)
    plt.bar(np.arange(n_rows), counter)
    print(n_rows)
    plt.xticks(np.arange(n_rows), unique_chords, rotation=0, fontsize=10)
    plt.savefig("figures/{}{} ({:d}).pdf".format(name,
                                                 SUFFIX, int(np.sum(counter))))


def print_result_histogram_chords(table, unique_chords, name="c_maj_a_min_transitions", text=False):
    columns = unique_chords + ["END"]

    row_sum = np.sum(table, axis=1)

    rows = []
    for i, c in enumerate(unique_chords):
        rows.append("{} ({:d})".format(c, int(row_sum[i])))
    n_transitions = np.sum(row_sum)
    print("Number of unique chords:", len(unique_chords))
    print("Number of transitions:", n_transitions)
    table = table / row_sum[:, None]

    # df = pd.DataFrame(table)
    # df["sum"] = pd.Series(row_sum)
    # print(df)

    plt.rcParams['figure.figsize'] = (20, 20)
    plt.matshow(table, cmap="jet")
    plt.xticks(np.arange(len(columns)), columns, rotation=90, fontsize=10)
    plt.yticks(np.arange(len(rows)), rows, fontsize=10)

    if text:
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                plt.text(j, i, "{:.1f}".format(
                    table[i][j]*100), ha="center", va="center", color="w")

    plt.savefig("figures/{}{} ({:d}).pdf".format(name, SUFFIX, int(n_transitions)))


def transpose_local_key(df):
    assert len(df["local_key"].unique()) == 1
    original_key = df["local_key"].iloc[0]
    is_minor_key = (original_key == original_key.lower())
    key = "i" if is_minor_key else "I"
    transposed_df = df.replace(
        to_replace={"local_key": original_key}, value=key, inplace=False)
    return transposed_df


def split_df(df, process_data_func, skip_short_phrases, skip_repetitions):
    prog_list = []
    mod_list = []
    start = 0
    for i in range(len(df)-1):
        if df["local_key"].iloc[i] != df["local_key"].iloc[i+1]:
            df_sub = df.iloc[start:i+1]
            mod = df.iloc[i:i+2]
            df_sub = transpose_local_key(df_sub)
            progression = process_data_func(df_sub)
            modulation = process_data_func(mod)
            prog_list.append(progression)

            mod_list.append(modulation.split(" "))
            start = i+1

    df_sub = df.iloc[start:len(df)]
    df_sub = transpose_local_key(df_sub)
    progression = process_data_func(df_sub)
    prog_list.append(progression)

    final_dataset = []
    for progression in prog_list:
        chord_list = progression.split(" ")
        if skip_repetitions:
            chord_list = [k for k, g in itertools.groupby(chord_list)]
        if len(chord_list) > skip_short_phrases:
            final_dataset.append(" ".join(chord_list))
    return final_dataset, mod_list


def get_dataset_and_modulation(all_csv, phrase_list, process_data_func, skip_short_phrases, skip_repetitions):
    df_all = pd.read_csv(all_csv)
    dataset = []
    modulation_list = []
    for beg, end in phrase_list:
        df = df_all[beg:end]

        if TRANSPOSE:
            df = transpose_phrase_to_c_maj_or_a_min(df)[0]
        prog_list, mod_list = split_df(
            df, process_data_func, skip_short_phrases, skip_repetitions)

        modulation_list += mod_list
        dataset += prog_list

    return dataset, modulation_list


def histogram_transposed_local_key_chords(chord_output, is_sorted=True):
    if is_sorted:
        process_data_func = dataframe_to_note_set_progression_sorted
    else:
        process_data_func = dataframe_to_note_set_progression

    phrase_list, _, _ = split_data_by_phrase(
        "data/phrases.txt", split_ratio=[1, 0, 0], shuffle=False)
    dataset, modulation_list = get_dataset_and_modulation(
        "data/all_annotations.csv", phrase_list, process_data_func,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS)

    unique_chords = set()
    progression_list = []
    for phrase in dataset:
        progression = phrase.split(" ")
        progression = convert_to_chord_name(
            progression, output=chord_output, is_sorted=is_sorted)
        progression = progression.split(" ")
        unique_chords.update(progression)
        progression_list.append(progression)

    unique_chords = list(unique_chords)
    unique_chords = sorted(unique_chords)
    n_rows = len(unique_chords)
    n_cols = n_rows + 1
    table = np.zeros((n_rows, n_cols))

    for progression in progression_list:
        for i in range(len(progression) - 1):
            start_chord = progression[i]
            end_chord = progression[i+1]
            start = unique_chords.index(start_chord)
            end = unique_chords.index(end_chord)
            table[start][end] += 1
        last = unique_chords.index(progression[-1])
        table[last][n_cols-1] += 1

    return table, unique_chords, modulation_list


def histogram_transposed_local_key_root():
    phrase_list, _, _ = split_data_by_phrase(
        "data/phrases.txt", split_ratio=[1, 0, 0], shuffle=False)
    dataset, modulation_list = get_dataset_and_modulation(
        "data/all_annotations.csv", phrase_list, dataframe_to_root_progression,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetitions=SKIP_REPETITIONS)

    unique_chords = set()
    progression_list = []
    for phrase in dataset:
        progression = phrase.split(" ")
        unique_chords.update(progression)
        progression_list.append(progression)

    unique_chords = list(unique_chords)
    unique_chords = sorted(unique_chords)
    n_rows = len(unique_chords)
    n_cols = n_rows + 1
    table = np.zeros((n_rows, n_cols))

    for progression in progression_list:
        for i in range(len(progression) - 1):
            start_chord = progression[i]
            end_chord = progression[i+1]
            start = unique_chords.index(start_chord)
            end = unique_chords.index(end_chord)
            table[start][end] += 1
        last = unique_chords.index(progression[-1])
        table[last][n_cols-1] += 1

    return table, unique_chords, modulation_list


def print_histogram_modulation(modulation_list, name="root_modulations", text=True):
    unique_chords = set()
    for mod in modulation_list:
        unique_chords.update(mod)

    unique_chords = list(unique_chords)
    unique_chords = sorted(unique_chords)
    n_rows = n_cols = len(unique_chords)
    table = np.zeros((n_rows, n_cols))

    for mod in modulation_list:
        if len(mod) != 2:
            continue
        start = unique_chords.index(mod[0])
        end = unique_chords.index(mod[1])
        table[start][end] += 1

    columns = unique_chords
    rows = unique_chords

    n_modulations = np.sum(table)
    print("Number of unique chords:", len(unique_chords))
    print("Number of modulations:", n_modulations)

    # df = pd.DataFrame(table)
    # df["sum"] = pd.Series(row_sum)
    # print(df)

    plt.rcParams['figure.figsize'] = (20, 20)
    plt.matshow(table, cmap="jet")
    plt.xticks(np.arange(len(columns)), columns, rotation=90, fontsize=10)
    plt.yticks(np.arange(len(rows)), rows, fontsize=10)

    if text:
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                plt.text(j, i, "{:d}".format(
                    int(table[i][j])), ha="center", va="center", color="w")

    plt.savefig("figures/{}{} ({:d}).pdf".format(name, SUFFIX, int(n_modulations)))


table, unique_chords, modulation_list = histogram_transposed_local_key_root()
print_result_histogram_chords(
    table, unique_chords, name="root_transitions", text=True)
print_histogram_modulation(modulation_list)

# print_histogram_root()


# table, unique_chords, modulation_list = histogram_transposed_local_key_chords("root_only", is_sorted=True)
# print_result_histogram_chords(
#     table, unique_chords, name="transposed_local_key_transitions_root_only_sorted", text=True)

# table, unique_chords = histogram_chords()
# print_result_histogram_chords(table, unique_chords)

# table, n_chords = histogram_numerals()
# print_result_histogram_numerals(table, n_chords)
