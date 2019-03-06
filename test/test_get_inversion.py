from src.preprocess_data import get_inversion
import pytest


@pytest.mark.parametrize("figbass,expected", [
    (None, [0, 1, 2]),
    ("6", [1, 2, 0]),
    ("64", [2, 0, 1]),

    ("7", [0, 1, 2, 3]),
    ("65", [1, 2, 3, 0]),
    ("43", [2, 3, 0, 1]),
    ("42", [3, 0, 1, 2]),
    ("2", [3, 0, 1, 2]),

    ("9", [0, 1, 2, 3, 4])
])
def test_get_inversion(figbass, expected):
    assert get_inversion(figbass) == expected
