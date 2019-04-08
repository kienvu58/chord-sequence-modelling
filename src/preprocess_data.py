from collections import Counter
import re
import numpy as np
import itertools


def get_key_number(key):
    key_list = ["c", "c#", "d", "d#", "e",
                "f", "f#", "g", "g#", "a", "a#", "b"]
    key = key.lower()
    key = "c" if key == "b#" else key
    key = "c#" if key == "db" else key
    key = "d#" if key == "eb" else key
    key = "e" if key == "fb" else key
    key = "f" if key == "e#" else key
    key = "f#" if key == "gb" else key
    key = "g#" if key == "ab" else key
    key = "a#" if key == "bb" else key
    key = "b" if key == "cb" else key

    return key_list.index(key)


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


def get_root(global_key, local_key, numeral, relativeroot=None):
    local_key_name = get_chord_name_from_key_and_numeral(global_key, local_key)
    if relativeroot is not None:
        local_key_name = get_chord_name_from_key_and_numeral(
            local_key_name, relativeroot)

    if numeral in ["Ger", "It", "Fr"]:
        return local_key_name

    root = get_chord_name_from_key_and_numeral(local_key_name, numeral)
    return root


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
    seventh_inversion_list = [[0, 1, 2, 3], [
        1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]

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


def convert_to_chord_name(progression, output="all", is_sorted=False):
    if not isinstance(progression, list):
        progression = progression.split(" ")
    chord_name_list = []
    for note_set_str in progression:
        chord_name = seperate_root(note_set_str.split("_"), output=output, is_sorted=is_sorted)
        chord_name_list.append(chord_name)
    return " ".join(chord_name_list)


def seperate_root(note_set, output="all", is_sorted=False):
    """
    cannot recovery the actual name of the note set
    """
    root_first_chords = {
        (0, 4, 7): "",
        (0, 3, 7): "m",
        (0, 3, 6): "o",
        (0, 4, 8): "+",
        (0, 3, 6, 10): "%7",
        (0, 3, 7, 10): "m7",
        (0, 3, 6, 9): "o7",
        (0, 4, 7, 10): "7",
        (0, 4, 7, 11): "M7",
        (0, 4, 8, 11): "+7",
        (0, 4, 7, 10, 1): "9",
    }

    root_second_chords = {
        (7, 0, 4): "64",
        (7, 0, 3): "m64",
        (6, 0, 3): "o64",
        (8, 0, 4): "+64",
        (10, 0, 3, 6): "%2",
        (10, 0, 3, 7): "m2",
        (9, 0, 3, 6): "o2",
        (10, 0, 4, 7): "2",
        (11, 0, 4, 7): "M2",
        (11, 0, 4, 8): "+2",
        (8, 0, 3, 6): "Ger",
        (8, 0, 6): "It",
        (8, 0, 2, 6): "Fr",
    }

    root_third_chords = {
        (4, 7, 0): "6",
        (3, 7, 0): "m6",
        (3, 6, 0): "o6",
        (4, 8, 0): "+6",
        (6, 10, 0, 3): "%43",
        (7, 10, 0, 3): "m43",
        (6, 9, 0, 3): "o43",
        (7, 10, 0, 4): "43",
        (7, 11, 0, 4): "M43",
        (8, 11, 0, 4): "+43",
    }

    root_forth_chords = {
        (3, 6, 10, 0): "%65",
        (3, 7, 10, 0): "m65",
        (3, 6, 9, 0): "o65",
        (4, 7, 10, 0): "65",
        (4, 7, 11, 0): "M65",
        (4, 8, 11, 0): "+65",
    }

    note_name_list = ["C", "C#", "D", "Eb", "E",
                      "F", "F#", "G", "Ab", "A", "Bb", "B"]

    note_set = np.array([int(note) for note in note_set])
    if not is_sorted:
        note_set_list = [note_set]
    else:
        note_set_list = list(itertools.permutations(note_set))

    possible_chord_list = []
    for note_set in note_set_list:
        if len(note_set) > 0:
            sub_root = tuple((note_set - note_set[0]) % 12)
            if sub_root in root_first_chords:
                possible_chord_list.append(
                    (note_name_list[note_set[0]], root_first_chords[sub_root]))

        if len(note_set) > 1:
            sub_root = tuple((note_set - note_set[1]) % 12)
            if sub_root in root_second_chords:
                possible_chord_list.append(
                    (note_name_list[note_set[1]], root_second_chords[sub_root]))

        if len(note_set) > 2:
            sub_root = tuple((note_set - note_set[2]) % 12)
            if sub_root in root_third_chords:
                possible_chord_list.append(
                    (note_name_list[note_set[2]], root_third_chords[sub_root]))

        if len(note_set) > 3:
            sub_root = tuple((note_set - note_set[3]) % 12)
            if sub_root in root_forth_chords:
                possible_chord_list.append(
                    (note_name_list[note_set[3]], root_forth_chords[sub_root]))

    if len(possible_chord_list) == 0:
        raise Exception("Unknown chords:", note_set)

    chord_list = []
    for chord in possible_chord_list:
        if output == "root_only":
            chord_list.append(chord[0])
        else:
            chord_list.append("{}{}".format(chord[0], chord[1]))
        

    possible_chord_list = sorted(chord_list)
    if output == "first":
        return possible_chord_list[0]
    elif output == "root_position":
        final_chord_list = []
        for chord in possible_chord_list:
            is_inversion = False
            for inv_str in ["6", "43", "65", "64", "2"]:
                if inv_str in chord:
                    is_inversion = True
                    break
            if not is_inversion:
                final_chord_list.append(chord)
        if len(final_chord_list) == 0:
            final_chord_list.append(possible_chord_list[0])
        return "/".join(final_chord_list)
    elif output == "root_only":
        most_common, _ = Counter(possible_chord_list).most_common(1)[0]
        return most_common
    else:
        return "/".join(possible_chord_list)


def get_note_set(quality, major, form, figbass, changes, use_changes=True):
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

    if use_changes:
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
