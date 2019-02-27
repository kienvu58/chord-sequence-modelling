from src.preprocess_data import get_chord_name
import pytest


@pytest.mark.parametrize("global_key,local_key,numeral,expected", [
    ("F", "I", "I", "F"),
    ("F", "I", "ii", "g"),
    ("F", "I", "iii", "a"),
    ("F", "I", "IV", "Bb"),
    ("F", "I", "V", "C"),
    ("F", "I", "vi", "d"),

    ("F", "v", "i", "c"),
    ("F", "v", "V", "G"),
    ("F", "v", "VI", "Ab"),

    ("F", "V", "I", "C"),
    ("F", "V", "ii", "d"),
    ("F", "V", "IV", "F"),
    ("F", "V", "V", "G"),
    ("F", "V", "vi", "a"),
    ("F", "V", "bVI", "Ab"),


    ("F", "III", "I", "A"),

    ("F", "IV", "I", "Bb"),
    ("F", "IV", "IV", "Eb"),

    ("F", "vi", "i", "d"),

    ("F", "ii", "i", "g"),
    ("F", "ii", "VI", "Eb"),

    ("F", "i", "i", "f"),
    ("F", "i", "VI", "Db"),

    ("F", "iv", "i", "bb"),
    ("F", "iv", "VI", "Gb"),

    ("F", "bVI", "I", "Db"),
    ("F", "bVI", "ii", "eb"),
    ("F", "bVI", "IV", "Gb"),
    ("F", "bVI", "V", "Ab"),
    ("F", "bVI", "vi", "bb"),

    ("F", "bII", "I", "Gb"),
    ("F", "bII", "V", "Db"),

    ("bb", "i", "i", "bb"),
    ("bb", "i", "iv", "eb"),
    ("bb", "i", "bII", "Cb"),
    ("bb", "#iii", "i", "d"),
    ("bb", "#iii", "V", "A"),
    ("bb", "I", "I", "Bb"),
    ("bb", "I", "V", "F"),

    ("g#", "i", "i", "g#"),
    ("g#", "i", "V", "D#"),
    ("g#", "i", "bII", "A"),
])
def test_get_chord_name_with_first_three_args(global_key, local_key, numeral, expected):
    assert get_chord_name(global_key, local_key, numeral) == expected

@pytest.mark.parametrize("global_key,local_key,numeral,relativeroot,expected", [
    ("Eb", "I", "Ger", "iii", "gGer6"),
    ("F", "i", "Ger", None, "fGer6"),
    ("G", "IV", "It", None, "CIt6"),
    ("C", "i", "Fr", None, "cFr6"),
])
def test_get_chord_name_with_special_chord(global_key, local_key, numeral, relativeroot, expected):
    assert get_chord_name(global_key, local_key, numeral, relativeroot=relativeroot) == expected


@pytest.mark.parametrize("global_key,local_key,numeral,relativeroot,expected", [
    ("Bb", "I", "V", "V", "C"),
    ("Bb", "I", "V", "IV", "Bb"),
    ("Bb", "I", "V", "ii", "G"),
    ("Bb", "I", "V", "vi", "D"),
    ("G", "IV", "V", "iii", "B"),
    ("Eb", "I", "vii", "IV", "g"),
    ("Eb", "I", "#vii", "vi", "b"),
])
def test_get_chord_name_with_relativeroot(global_key, local_key, numeral, relativeroot, expected):
    assert get_chord_name(global_key, local_key, numeral, relativeroot=relativeroot) == expected


@pytest.mark.parametrize("global_key,local_key,numeral,form,figbass,expected", [
    ("C", "I", "I", "M", "7", "CM7"),
    ("C", "I", "I", "M", "43", "CM43"),
    ("C", "I", "I", "M", "2", "CM2"),
    ("C", "I", "vii", "o", "65", "bo65"),
    ("C", "I", "vii", "o", "6", "bo6"),
    ("C", "I", "I", "+", "64", "C+64"),
    ("C", "I", "I", "+", None, "C+"),
    ("C", "I", "vii", "%", "7", "b%7"),
    ("C", "I", "V", None, "7", "G7"),
])
def test_get_chord_name_with_form_and_figbass(global_key, local_key, numeral, form, figbass, expected):
    assert get_chord_name(global_key, local_key, numeral, form=form, figbass=figbass) == expected