import re
import numpy as np


def is_key_with_sharp(key):
    key_with_sharp_list = ["A", "B", "C", "D", "E", "G", "a", "b", "e"]
    key_with_flat_list = ["F", "c", "d", "f", "g"]
    if len(key) == 1:
        if key in key_with_sharp_list:
            with_sharp = True
        elif key in key_with_flat_list:
            with_sharp = False
        else:
            raise ValueError("Not supported key {}".format(key))
    elif len(key) == 2:
        if key[1] == "#":
            with_sharp = True
        elif key[1] == "b":
            with_sharp = False
        else:
            raise ValueError("Not supported key {}".format(key))
    return with_sharp


def parse_numeral(numeral):
    accidental = numeral.count("#") - numeral.count("b")

    numeral = numeral.replace("#", "")
    numeral = numeral.replace("b", "")
    major_quality = (numeral == numeral.upper())

    numeral = numeral.upper()
    scale_degree_list = ["I", "II", "III", "IV", "V", "VI", "VII"]
    scale_degree = scale_degree_list.index(numeral) + 1

    return scale_degree, accidental, major_quality


def get_chord_name_from_key_and_numeral(key, numeral):
    chromatic_scale_with_sharp = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    chromatic_scale_with_flat = ["C", "Db", "D",
                                 "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    chromatic_scale_with_enharmonic_sharp = [
        "B#", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"]
    chromatic_scale_with_enharmonic_flat = [
        "C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"]

    major_scale = [0, 2, 4, 5, 7, 9, 11]
    minor_scale = [0, 2, 3, 5, 7, 8, 10]

    minor_mode = (key == key.lower())
    diatonic_scale = minor_scale if minor_mode else major_scale

    with_sharp = is_key_with_sharp(key)
    key_lower = key.lower()
    if with_sharp:
        if key_lower in ["b#", "e#"]:
            chromatic_scale = chromatic_scale_with_enharmonic_sharp
        else:
            chromatic_scale = chromatic_scale_with_sharp
    else:
        if key_lower in ["cb", "fb"]:
            chromatic_scale = chromatic_scale_with_enharmonic_flat
        else:
            chromatic_scale = chromatic_scale_with_flat
    chromatic_scale_lower = [note.lower() for note in chromatic_scale]

    scale_degree, accidental, major_quality = parse_numeral(numeral)
    key_position = chromatic_scale_lower.index(key_lower)
    chord_position = (key_position + diatonic_scale[scale_degree-1]) % 12
    chord_name = chromatic_scale[chord_position]

    if accidental == 1 and len(chord_name) == 1:
        chord_name += "#"
    elif accidental == -1 and len(chord_name) == 1:
        chord_name += "b"
    else:
        chord_position = (chord_position + accidental) % 12
        chord_name = chromatic_scale[chord_position]

    chord_name = chord_name if major_quality else chord_name.lower()

    return chord_name


def get_chord_name(global_key, local_key, numeral, form=None, figbass=None, relativeroot=None):
    # mode_list = ["maj", "min"]
    # root_list = ["Cb", "C", "C#", "Db", "D", "D#", "Eb", "E", "E#", "Fb", "F", "F#",
    #              "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B", "B#"]
    # form_list = ["M", "%", "o", "+"]
    # figbass_list = ["6", "64", "7", "65", "43", "2"]

    if numeral not in ["Ger", "It", "Fr"]:
        local_key_name = get_chord_name_from_key_and_numeral(
            global_key, local_key)
        if relativeroot is not None:
            relativeroot_name = get_chord_name_from_key_and_numeral(
                local_key_name, relativeroot)
            chord_name = get_chord_name_from_key_and_numeral(
                relativeroot_name, numeral)
        else:
            chord_name = get_chord_name_from_key_and_numeral(
                local_key_name, numeral)

        if form is not None:
            chord_name += form

        if figbass is not None:
            chord_name += figbass
    else:
        local_key_name = get_chord_name_from_key_and_numeral(
            global_key, local_key)
        if relativeroot is not None:
            local_key_name = get_chord_name_from_key_and_numeral(
                local_key_name, relativeroot)

        chord_name = local_key_name + numeral + "6"

    return chord_name


def parse_chord_name(chord_name):
    """
    form_name_list = [
        "major triad",
        "minor triad",
        "diminished triad",
        "augmented triad",

        "major seventh",
        "minor seventh",
        "dominant seventh",
        "diminished seventh",
        "half-diminised seventh",
        "augmented seventh",

        "German sixth",
        "Italian sixth",
        "French sixth",
    ]
    """

    figbass_name_list = [
        "root",
        "first",
        "second",
        "third",
    ]

    seventh_figbass_list = ["7", "65", "43", "2"]
    triad_figbass_list = [None, "6", "64"]

    pattern = re.compile(r"""(?P<key>[a-gA-G](b|\#)?)
                             (?P<form>([%o+M]|Ger6|It6|Fr6))?
                             (?P<figbass>(7|65|43|2|64|6))?
                            """, re.VERBOSE)

    match = pattern.match(chord_name)
    key = match.group("key")
    form = match.group("form")
    figbass = match.group("figbass")

    key_is_lowercase = (key == key.lower())

    if form == "Ger6":
        return key, "German sixth", figbass_name_list[0]
    elif form == "It6":
        return key, "Italian sixth", figbass_name_list[0]
    elif form == "Fr6":
        return key, "French sixth", figbass_name_list[0]
    else:
        if figbass in seventh_figbass_list:
            if form == "%":
                form_name = "half-diminished seventh"
            elif form == "o":
                form_name = "diminished seventh"
            elif form == "M":
                form_name = "major seventh"
            elif form == "+":
                form_name = "augmented seventh"
            elif key_is_lowercase:
                form_name = "minor seventh"
            else:
                form_name = "dominant seventh"
            figbass_name = figbass_name_list[seventh_figbass_list.index(
                figbass)]
        else:
            if form == "o":
                form_name = "diminished triad"
            elif form == "+":
                form_name = "augmented triad"
            elif key_is_lowercase:
                form_name = "minor triad"
            else:
                form_name = "major triad"
            figbass_name = figbass_name_list[triad_figbass_list.index(
                figbass)]

    return key, form_name, figbass_name

def get_inversion(figbass):
    if figbass == "42":
        figbass = "2"
    triad_figbass_list = [None, "6", "64"]
    triad_inversion_list = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]

    seventh_figbass_list = ["7", "65", "43", "2"]
    seventh_inversion_list = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]

    if figbass in triad_figbass_list:
        index = triad_figbass_list.index(figbass)
        return triad_inversion_list[index]
    elif figbass in seventh_figbass_list:
        index = seventh_figbass_list.index(figbass)
        return seventh_inversion_list[index]
    elif figbass == "9":
        return [0, 1, 2, 3, 4]
    else:
        raise ValueError("Unknown figured bass:", figbass)

def get_note_set(quality, major, form, figbass, changes):
    """
        quality: major | minor | Ger | It | Fr derived from numeral
        major: True | False whether local key is major or minor
    """
    inversion = get_inversion(figbass)

    if quality == "Ger":
        return [8, 0, 3, 6]
    elif quality == "It":
        return [8, 0, 6]
    elif quality == "Fr":
        return [8, 0, 2, 6]
    # triad
    elif len(inversion) == 3:
        if quality == "minor":
            if form is None:
                # minor triad
                note_set = [0, 3, 7]
            elif form == "o":
                # diminished triad
                note_set = [0, 3, 6]
            elif form == "+":
                # augmented triad
                note_set = [0, 4, 8]
            elif form == "%":
                # half-diminished seventh
                note_set = [0, 3, 6, 10]
                inversion = [0, 1, 2, 3]
            else:
                raise ValueError("Unknown chord: ", quality, form, inversion)
        elif quality == "major":
            if form is None:
                # major triad
                note_set = [0, 4, 7]
            elif form == "+":
                # augmented triad
                note_set = [0, 4, 8]
            else: 
                raise ValueError("Unknown chord: ", quality, form, inversion)
        else:
            raise ValueError("Unknown chord: ", quality, form, inversion)
    # seventh
    elif len(inversion) == 4:
        if quality == "minor":
            if form is None:
                # minor seventh
                note_set = [0, 3, 7, 10]
            elif form == "o":
                # diminished seventh
                note_set = [0, 3, 6, 9]
            elif form == "%":
                # half-diminished seventh
                note_set = [0, 3, 6, 10]
            else:
                raise ValueError("Unknown chord: ", quality, form, inversion)
        elif quality == "major":
            if form is None:
                # dominant seventh
                note_set = [0, 4, 7, 10]
            elif form == "M":
                # major seventh
                note_set = [0, 4, 7, 11]
            elif form == "+":
                # augmented seventh
                note_set = [0, 4, 8, 11]
            else:
                raise ValueError("Unknown chord: ", quality, form, inversion)
        else:
            raise ValueError("Unknown chord: ", quality, form, inversion)
    # nineth
    elif len(inversion) == 5:
        # dominant minor ninth
        note_set = [0, 4, 7, 10, 1]
    else:
        raise ValueError("Unknown chord: ", quality, form, inversion)

    note_set = apply_changes(note_set, changes, major) 
    note_set = apply_inversion(note_set, inversion)

    return note_set

def apply_inversion(note_set, inversion):
    inversion_size = len(inversion)
    first_part = list(np.array(note_set[:inversion_size])[inversion])
    second_part = note_set[inversion_size:]
    note_set = first_part + second_part
    return note_set

def apply_changes(note_set, changes, major):
    # major_scale = [0, 2, 4, 5, 7, 9, 11]
    # minor_scale = [0, 2, 3, 5, 7, 8, 10]
    change_list = parse_changes(changes)
    for change in change_list:
        if change == "2":
            root_index = note_set.index(0)
            note_set[root_index] = 2
        elif change == "4":
            if 3 in note_set:
                third_index = note_set.index(3)
            elif 4 in note_set:
                third_index = note_set.index(4)
            note_set[third_index] = 5
        elif change == "6":
            if 7 in note_set:
                fifth_index = note_set.index(7)
            else:
                fifth_index = note_set.index(6)
            if 4 in note_set and major:
                note_set[fifth_index] = 9
            else:
                note_set[fifth_index] = 8
        elif change in ["9", "+9", "+2", "+#2"]:
            add_note(note_set, 2)
        elif change == "+4":
            add_note(note_set, 5)
        elif change == "+6":
            if 4 in note_set and major:
                add_note(note_set, 9)
            else:
                add_note(note_set, 8)
        elif change in ["7", "#7"]:
            add_note(note_set, 11)
        elif change == "#2":
            if 4 in note_set:
                third_index = note_set.index(4)
                note_set[third_index] = 3
        elif change == "#4":
            if 7 in note_set:
                fifth_index = note_set.index(7)
                note_set[fifth_index] = 6
        elif change == "b6":
            if 7 in note_set:
                fifth_index = note_set.index(7)
                note_set[fifth_index] = 8
        elif change in ["b9", "+b9"]:
            add_note(note_set, 1)
        elif change == "b2":
            if 0 in note_set:
                root_index = note_set.index(0)
                note_set[root_index] = 1
        elif change == "+b2":
            add_note(note_set, 1)
        elif change == "b4":
            if 3 in note_set:
                third_index = note_set.index(3)
                note_set[third_index] = 4
            elif 4 in note_set:
                third_index = note_set.index(4)
                note_set[third_index] = 5
        elif change == "#5":
            if 7 in note_set:
                fifth_index = note_set.index(7)
                note_set[fifth_index] = 8
        else:
            pass

    return note_set

def add_note(note_set, note):
    if note not in note_set:
        note_set.append(note)

def parse_changes(changes):
    if changes:
        matches = re.findall(r"(\+)?(b|#)?(2|4|5|6|7|9|11|13)", changes)
        matches = ["".join(m) for m in matches]
        return matches
    return []