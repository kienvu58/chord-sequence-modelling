from src.preprocess_data import parse_numeral
import pytest


@pytest.mark.parametrize("numeral,expected", [
    ("I", (1, 0, True)),
    ("II", (2, 0, True)),
    ("III", (3, 0, True)),
    ("IV", (4, 0, True)),
    ("V", (5, 0, True)),
    ("VI", (6, 0, True)),
    ("VII", (7, 0, True)),

    ("i", (1, 0, False)),
    ("ii", (2, 0, False)),
    ("iii", (3, 0, False)),
    ("iv", (4, 0, False)),
    ("v", (5, 0, False)),
    ("vi", (6, 0, False)),
    ("vii", (7, 0, False)),

    ("#iv", (4, 1, False)),
    ("#vii", (7, 1, False)),
    ("#III", (3, 1, True)),

    ("biv", (4, -1, False)),
    ("bII", (2, -1, True)),

    ("##i", (1, 2, False)),
    ("bbII", (2, -2, True)),
])
def test_parse_numeral(numeral, expected):
    assert parse_numeral(numeral) == expected