import torch
import types
from allennlp.data.vocabulary import Vocabulary
from modules.data_preprocessors import (
    parse_chord_name_core,
    convert_to_note_set,
    convert_to_note_set_core,
    get_key_number,
)

# match exactly        10 points (key +5, form +1, figbass +1, 2notes +1, 3notes +2, (4notes +2))    total: 10 + 10 + 2
# match key             5 points
# match form            1 points
# match figbass         1 points
# match enharmonic key  4 points
# match 2 notes         1 points
# match 3 notes         2 points (2notes +1)                                                         total: 2 + 1
# match 4 notes         2 points (2notes +1, 3notes +2)                                              total: 2 + 3


class Score:
    score = {
        "exactly": 20,
        "key": 5,
        "form": 1,
        "figbass": 1,
        "enharmonic_key": 4,
        "2_notes": 1,
        "3_notes": 2,
        "4_notes": 2,
    }

    @staticmethod
    def match_exactly(gold, pred):
        if gold == pred:
            return Score.score["exactly"]
        return 0

    @staticmethod
    def match_key(gold, pred):
        gold_key, _, _ = parse_chord_name_core(gold)
        pred_key, _, _ = parse_chord_name_core(pred)
        if gold_key is not None and gold_key == pred_key:
            return Score.score["key"]
        return 0

    @staticmethod
    def match_triad_form(gold, pred):
        _, gold_note_set = convert_to_note_set_core(gold)
        _, pred_note_set = convert_to_note_set_core(pred)
        if gold_note_set is not None and pred_note_set is not None and gold_note_set[:3] == pred_note_set[:3]:
            return Score.score["form"]
        return 0

    @staticmethod
    def match_figbass(gold, pred):
        gold_key, _, gold_figbass = parse_chord_name_core(gold)
        pred_key, _, pred_figbass = parse_chord_name_core(pred)
        if gold_key is not None and pred_key is not None and gold_figbass == pred_figbass:
            return Score.score["figbass"]
        return 0

    @staticmethod
    def match_enharmonic_key(gold, pred):
        gold_key, _, _ = parse_chord_name_core(gold)
        pred_key, _, _ = parse_chord_name_core(pred)
        if gold_key is not None and pred_key is not None and gold_key != pred_key and get_key_number(gold_key) == get_key_number(pred_key):
            return Score.score["enharmonic_key"]
        return 0

    @staticmethod
    def match_2_notes(gold, pred):
        gold_note_set = convert_to_note_set(gold)
        pred_note_set = convert_to_note_set(pred)
        intersection = set([note for note in gold_note_set if note in pred_note_set])
        if len(intersection) >= 2:
            return Score.score["2_notes"]
        return 0

    @staticmethod
    def match_3_notes(gold, pred):
        gold_note_set = convert_to_note_set(gold)
        pred_note_set = convert_to_note_set(pred)
        intersection = set([note for note in gold_note_set if note in pred_note_set])
        if len(intersection) >= 3:
            return Score.score["3_notes"]
        return 0

    @staticmethod
    def match_4_notes(gold, pred):
        gold_note_set = convert_to_note_set(gold)
        pred_note_set = convert_to_note_set(pred)
        intersection = set([note for note in gold_note_set if note in pred_note_set])
        if len(intersection) >= 4:
            return Score.score["4_notes"]
        return 0


def test_match_functions():
    assert Score.match_exactly("AbM7", "AbM7") != 0
    assert Score.match_exactly("@end@", "@end@") != 0
    assert Score.match_key("FbM7", "Fbm") != 0
    assert Score.match_key("FbM7", "FM7") == 0
    assert Score.match_key("@end@", "FM7") == 0
    assert Score.match_triad_form("Co7", "Fo") != 0
    assert Score.match_triad_form("C7", "CM7") != 0
    assert Score.match_triad_form("Cm7", "CM7") == 0
    assert Score.match_figbass("CGer6", "FIt6") != 0
    assert Score.match_figbass("F7", "Cm7") != 0
    assert Score.match_figbass("F", "Cm") != 0
    assert Score.match_figbass("@end@", "Cm") == 0
    assert Score.match_enharmonic_key("G#", "Ab") != 0
    assert Score.match_enharmonic_key("C", "B#") != 0
    assert Score.match_enharmonic_key("B#", "B#") == 0
    assert Score.match_enharmonic_key("G", "A") == 0
    assert Score.match_2_notes("C", "Am") != 0
    assert Score.match_2_notes("C", "Cm") != 0
    assert Score.match_2_notes("C", "F") == 0
    assert Score.match_3_notes("C", "C7") != 0
    assert Score.match_3_notes("C+", "Ab+") != 0
    assert Score.match_3_notes("C", "Cm") == 0
    assert Score.match_4_notes("Bbo7", "Eo7") != 0
    assert Score.match_4_notes("AbGer6", "E7") != 0
    assert Score.match_4_notes("C7", "E7") == 0


def get_target_distribution(gold_token, vocab):
    vocab_size = vocab.get_vocab_size()
    gold_index = vocab.get_token_index(gold_token)

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

    weight /= weight.sum()
    return weight


def create_transformer_weight():
    vocab = Vocabulary().from_files("data/vocabulary")

    token_weight_list = []
    for index, token in vocab.get_index_to_token_vocabulary().items():
        token_weight = get_target_distribution(token, vocab)
        token_weight_list.append(token_weight)

    weight = torch.stack(token_weight_list)
    torch.save(weight, "data/transformer_weight.th")


create_transformer_weight()

