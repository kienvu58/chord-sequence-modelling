
# convert dataset into phrases
#   no augmented
#   transpose to C major or C minor key

# create a table of transition
#   count the number of each transition
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
                             transpose_phrase_to_c_maj_or_a_min)

SHORT_PHRASE_LEN = 1
SKIP_REPETIONS = True


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

    plt.savefig("figures/diatonic_scale_transitions ({:.1f}%).pdf".format(n_transitions/(n_chords-1)*100))


def histogram_chords():
    is_sorted = True
    if is_sorted:
        process_data_func = dataframe_to_note_set_progression_sorted
    else:
        process_data_func = dataframe_to_note_set_progression

    phrase_list, _, _ = split_data_by_phrase(
        "data/phrases.txt", split_ratio=[1, 0, 0], shuffle=False)
    dataset = get_dataset(
        "data/all_annotations.csv", phrase_list, process_data_func,
        skip_short_phrases=SHORT_PHRASE_LEN, skip_repetions=SKIP_REPETIONS, augment=True, augment_func=transpose_phrase_to_c_maj_or_a_min)

    unique_chords = set()
    progression_list = []
    for phrase in dataset:
        progression = phrase.split(" ")
        progression = convert_to_chord_name(progression, output="root", is_sorted=is_sorted)
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


def print_result_histogram_chords(table, unique_chords):
    columns = unique_chords + ["END"]
    rows = unique_chords

    row_sum = np.sum(table, axis=1)
    n_transitions = np.sum(row_sum)
    print("Number of transitions:", n_transitions)
    table = table / row_sum[:, None]

    # df = pd.DataFrame(table)
    # df["sum"] = pd.Series(row_sum)
    # print(df)

    plt.rcParams['figure.figsize'] = (20, 20)
    plt.matshow(table, cmap="jet")
    plt.xticks(np.arange(len(columns)), columns, rotation=90, fontsize=10)
    plt.yticks(np.arange(len(rows)), rows, fontsize=10)

    # for i in range(table.shape[0]):
    #     for j in range(table.shape[1]):
    #         plt.text(j, i, "{:.1f}".format(
    #             table[i][j]*100), ha="center", va="center", color="w")

    plt.savefig("figures/c_maj_a_min_transitions ({:d}).pdf".format(int(n_transitions)))


table, unique_chords = histogram_chords()
print_result_histogram_chords(table, unique_chords)

# table, n_chords = histogram_numerals()
# print_result_histogram_numerals(table, n_chords)
