from src.preprocess_data import parse_chord_name
import pytest


@pytest.mark.parametrize("chord_name,expected", [
    ("Eb", ("Eb", "major triad", "root")),
    ("Eb6", ("Eb", "major triad", "first")),
    ("Eb64", ("Eb", "major triad", "second")),

    ("c", ("c", "minor triad", "root")),
    ("f6", ("f", "minor triad", "first")),
    ("c64", ("c", "minor triad", "second")),

    ("Ab+", ("Ab", "augmented triad", "root")),
    ("Eb+6", ("Eb", "augmented triad", "first")),
    ("Eb+64", ("Eb", "augmented triad", "second")),

    ("do", ("d", "diminished triad", "root")),
    ("go6", ("g", "diminished triad", "first")),
    ("ao64", ("a", "diminished triad", "second")),

    ("BbM7", ("Bb", "major seventh", "root")),
    ("AbM65", ("Ab", "major seventh", "first")),
    ("AbM43", ("Ab", "major seventh", "second")),
    ("CM2", ("C", "major seventh", "third")),

    ("f7", ("f", "minor seventh", "root")),
    ("db65", ("db", "minor seventh", "first")),
    ("f43", ("f", "minor seventh", "second")),
    ("f2", ("f", "minor seventh", "third")),

    ("Bb7", ("Bb", "dominant seventh", "root")),
    ("Bb65", ("Bb", "dominant seventh", "first")),
    ("Bb43", ("Bb", "dominant seventh", "second")),
    ("Bb2", ("Bb", "dominant seventh", "third")),

    ("bo7", ("b", "diminished seventh", "root")),
    ("f#o65", ("f#", "diminished seventh", "first")),
    ("f#o43", ("f#", "diminished seventh", "second")),
    ("f#o2", ("f#", "diminished seventh", "third")),

    ("d%7", ("d", "half-diminished seventh", "root")),
    ("f#%65", ("f#", "half-diminished seventh", "first")),
    ("f#%43", ("f#", "half-diminished seventh", "second")),
    ("d#%2", ("d#", "half-diminished seventh", "third")),

    ("C+7", ("C", "augmented seventh", "root")),
    ("Ab+65", ("Ab", "augmented seventh", "first")),
    ("Ab+43", ("Ab", "augmented seventh", "second")),
    ("Ab+2", ("Ab", "augmented seventh", "third")),

    ("fGer6", ("f", "German sixth", "root")),
    ("DIt6", ("D", "Italian sixth", "root")),
    ("EFr6", ("E", "French sixth", "root")),
])
def test_parse_chord_name(chord_name, expected):
    assert parse_chord_name(chord_name) == expected