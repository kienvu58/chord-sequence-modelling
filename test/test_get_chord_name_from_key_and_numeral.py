from src.preprocess_data import get_chord_name_from_key_and_numeral
import pytest


@pytest.mark.parametrize("key,numeral,expected", [
    ("C", "I", "C"),
    ("C", "ii", "d"),
    ("C", "iii", "e"),
    ("C", "IV", "F"),
    ("C", "V", "G"),
    ("C", "vi", "a"),
    ("C", "vii", "b"),

    ("F#", "I", "F#"),
    ("F#", "ii", "g#"),
    ("F#", "V", "C#"),

    ("Ab", "I", "Ab"),
    ("Ab", "iii", "c"),
    ("Ab", "IV", "Db"),

    ("bb", "i", "bb"),
    ("bb", "III", "Db"),
    ("bb", "v", "f"),
    ("bb", "bII", "Cb"),
    ("bb", "bV", "Fb"),
    ("bb", "VI", "Gb"),

    ("a", "i", "a"),
    ("a", "v", "e"),
    ("a", "III", "C"),

    ("f", "VI", "Db"),
    ("F#", "#iv", "b#"),
    ("F#", "#vii", "f#"),
])
def test_get_chord_name_from_numeral_and_key(key, numeral, expected):
    assert get_chord_name_from_key_and_numeral(key, numeral) == expected
