
# convert dataset into phrases
#   no augmented
#   transpose to C major or C minor key

# create a table of transition
#   count the number of each transition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def histogram_numeral():
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
    
def print_result_histogram_numeral(table, n_chords):
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
            plt.text(j, i, "{:.1f}".format(table[i][j]*100), ha="center", va="center", color="w")

    plt.show()

table, n_chords = histogram_numeral()
print_result_histogram_numeral(table, n_chords)