from src.preprocess_data import get_note_set
import pytest


@pytest.mark.parametrize("quality,major,form,figbass,changes,expected", [
    # dominant minor ninth
    ("major", True, None, "9", None, [0, 4, 7, 10, 1]),

    # major triad
    ("major", True, None, None, None, [0, 4, 7]),
    ("major", True, None, "6", None, [4, 7, 0]),
    ("major", True, None, "64", None, [7, 0, 4]),

    # augmented triad
    ("major", True, "+", None, None, [0, 4, 8]),
    ("major", True, "+", "6", None, [4, 8, 0]),
    ("major", True, "+", "64", None, [8, 0, 4]),

    # minor triad
    ("minor", True, None, None, None, [0, 3, 7]),
    ("minor", True, None, "6", None, [3, 7, 0]),
    ("minor", True, None, "64", None, [7, 0, 3]),

    # diminished triad
    ("minor", True, "o", None, None, [0, 3, 6]),
    ("minor", True, "o", "6", None, [3, 6, 0]),
    ("minor", True, "o", "64", None, [6, 0, 3]),

    # dominant seventh
    ("major", True, None, "7", None, [0, 4, 7, 10]),
    ("major", True, None, "65", None, [4, 7, 10, 0]),
    ("major", True, None, "43", None, [7, 10, 0, 4]),
    ("major", True, None, "2", None, [10, 0, 4, 7]),

    # major seventh
    ("major", True, "M", "7", None, [0, 4, 7, 11]),

    # minor seventh
    ("minor", True, None, "7", None, [0, 3, 7, 10]),

    # diminished seventh
    ("minor", True, "o", "7", None, [0, 3, 6, 9]),

    # half-diminished seventh
    ("minor", True, "%", "7", None, [0, 3, 6, 10]),

    # augmented seventh
    ("major", True, "+", "7", None, [0, 4, 8, 11]),

    # changes 64
    ("major", True, None, None, "64", [0, 5, 9]),
    ("major", False, None, None, "64", [0, 5, 8]),
    ("major", True, None, "7", "64", [0, 5, 9, 10]),
    ("major", False, None, "7", "64", [0, 5, 8, 10]),

    ("major", False, None, None, "4", [0, 5, 7]),
    ("major", True, None, "7", "4", [0, 5, 7, 10]),

    ("major", False, None, None, "2", [2, 4, 7]),
    ("major", True, None, "7", "2", [2, 4, 7, 10]),

    ("major", True, None, None, "6", [0, 4, 9]),
    ("major", False, None, None, "6", [0, 4, 8]),
    ("minor", True, None, None, "6", [0, 3, 8]),
    ("major", True, None, "7", "6", [0, 4, 9, 10]),
    ("major", False, None, "7", "6", [0, 4, 8, 10]),

    ("major", True, None, None, "9", [0, 4, 7, 2]),
    ("major", False, None, None, "9", [0, 4, 7, 2]),
    ("minor", True, None, None, "9", [0, 3, 7, 2]),
    ("major", True, None, "7", "9", [0, 4, 7, 10, 2]),
    ("major", False, None, "7", "9", [0, 4, 7, 10, 2]),

    ("major", False, None, None, "7", [0, 4, 7, 11]),
])
def test_get_note_set(quality, major, form, figbass, changes, expected):
    assert get_note_set(quality, major, form, figbass, changes) == expected
