import torch
import types
import itertools
from allennlp.data.vocabulary import Vocabulary
from modules.data_preprocessors import (
    parse_chord_name_core,
    convert_to_note_set,
    convert_to_note_set_core,
    get_key_number,
)


class Score:
    score = {
        "token_name": 5,
        "key_name": 3,
        "key_number": 3,
        "triad_form": 3,
        "figbass": 1,
        "note_pair": 3,
    }

    @staticmethod
    def match_token_name(gold, pred):
        if gold == pred:
            return Score.score["token_name"]
        return 0

    @staticmethod
    def match_key_name(gold, pred):
        gold_key, _, _ = parse_chord_name_core(gold)
        pred_key, _, _ = parse_chord_name_core(pred)
        if gold_key is not None and gold_key == pred_key:
            return Score.score["key_name"]
        return 0

    @staticmethod
    def match_triad_form(gold, pred):
        _, gold_note_set = convert_to_note_set_core(gold)
        _, pred_note_set = convert_to_note_set_core(pred)
        if (
            gold_note_set is not None
            and pred_note_set is not None
            and gold_note_set[:3] == pred_note_set[:3]
        ):
            return Score.score["triad_form"]
        return 0

    @staticmethod
    def match_figbass(gold, pred):
        gold_key, _, gold_figbass = parse_chord_name_core(gold)
        pred_key, _, pred_figbass = parse_chord_name_core(pred)
        if (
            gold_key is not None
            and pred_key is not None
            and gold_figbass == pred_figbass
        ):
            return Score.score["figbass"]
        return 0

    @staticmethod
    def match_key_number(gold, pred):
        gold_key, _, _ = parse_chord_name_core(gold)
        pred_key, _, _ = parse_chord_name_core(pred)
        if (
            gold_key is not None
            and pred_key is not None
            and get_key_number(gold_key) == get_key_number(pred_key)
        ):
            return Score.score["key_number"]
        return 0

    @staticmethod
    def match_note_pair(gold, pred):
        gold_note_set = convert_to_note_set(gold)
        pred_note_set = convert_to_note_set(pred)
        gold_note_pair_set = get_note_pair_set(gold_note_set)
        pred_note_pair_set = get_note_pair_set(pred_note_set)
        intersection = set(
            [note for note in gold_note_pair_set if note in pred_note_pair_set]
        )
        return len(intersection) * Score.score["note_pair"]


def get_note_pair_set(note_set):
    pair_set = list(itertools.product(note_set, note_set))
    pair_set = set(
        [
            "_".join([str(note) for note in sorted(pair)])
            for pair in pair_set
            if pair[0] != pair[1]
        ]
    )
    return pair_set


get_note_pair_set([0, 4, 7, 10])


def test_match_functions():
    assert Score.match_token_name("AbM7", "AbM7") != 0
    assert Score.match_token_name("@end@", "@end@") != 0
    assert Score.match_key_name("FbM7", "Fbm") != 0
    assert Score.match_key_name("FbM7", "FM7") == 0
    assert Score.match_key_name("@end@", "FM7") == 0
    assert Score.match_triad_form("Co7", "Fo") != 0
    assert Score.match_triad_form("C7", "CM7") != 0
    assert Score.match_triad_form("Cm7", "CM7") == 0
    assert Score.match_figbass("CGer6", "FIt6") != 0
    assert Score.match_figbass("F7", "Cm7") != 0
    assert Score.match_figbass("F", "Cm") != 0
    assert Score.match_figbass("@end@", "Cm") == 0
    assert Score.match_key_number("G#", "Ab") != 0
    assert Score.match_key_number("C", "B#") != 0
    assert Score.match_key_number("B#", "B#") != 0
    assert Score.match_key_number("G", "A") == 0
    assert Score.match_note_pair("C", "C") == Score.score["note_pair"] * 3
    assert Score.match_note_pair("C", "Cm") == Score.score["note_pair"]
    assert Score.match_note_pair("C", "C7") == Score.score["note_pair"] * 3
    assert Score.match_note_pair("C7", "C7") == Score.score["note_pair"] * 6


test_match_functions()


def get_target_distribution(gold_token, vocab):
    vocab_size = vocab.get_vocab_size()

    weight = torch.zeros((vocab_size,), dtype=torch.float)
    for index in range(vocab_size):
        token = vocab.get_token_from_index(index)
        match_func_list = [func for func in dir(Score) if func.startswith("match")]
        score = sum(
            [
                getattr(Score, func_name)(gold_token, token)
                for func_name in match_func_list
            ]
        )
        weight[index] = score

    # weight /= weight.sum()
    return weight


def create_target_weight():
    vocab = Vocabulary().from_files("data/vocabulary")

    token_weight_list = []
    for index, token in vocab.get_index_to_token_vocabulary().items():
        token_weight = get_target_distribution(token, vocab)
        token_weight_list.append(token_weight)

    weight = torch.stack(token_weight_list)
    s = Score.score
    torch.save(
        weight,
        "data/targets/target_{}{}{}{}{}{}.th".format(
            s["token_name"],
            s["key_name"],
            s["key_number"],
            s["triad_form"],
            s["figbass"],
            s["note_pair"],
        ),
    )


# import matplotlib.pyplot as plt
# import numpy as np

# def transform():
#     vocab = Vocabulary().from_files("data/vocabulary")
#     token = "A+"
#     token_weight = get_target_distribution(token, vocab)
#     # for i, t in vocab.get_index_to_token_vocabulary().items():
#     #     print(t, token_weight[i].item())

#     plt.tick_params(
#         axis="both",
#         left=False,
#         top=False,
#         right=False,
#         bottom=False,
#         labelleft=True,
#         labeltop=False,
#         labelright=False,
#         labelbottom=False,
#     )
#     ax = plt.gca()
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_visible(True)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(True)
#     x_pos = np.arange(vocab.get_vocab_size())
#     plt.bar(x_pos, token_weight.numpy(), width=1.0)
#     plt.show()


create_target_weight()
# transform()

