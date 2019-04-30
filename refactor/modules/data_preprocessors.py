def transpose_to_all_keys(df):
    assert len(df["global_key"].unique()) == 1
    original_key = df["global_key"].iloc[0]
    is_minor_key = original_key == original_key.lower()
    major_key_list = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
    minor_key_list = ["a", "e", "b", "f#", "c#", "g#", "eb", "bb", "f", "c", "g", "d"]
    key_list = minor_key_list if is_minor_key else major_key_list
    transposed_list = []
    for key in key_list:
        transposed_df = df.replace(
            to_replace={"global_key": original_key}, value=key, inplace=False
        )
        transposed_list.append(transposed_df)

    return transposed_list


def get_chord_name(
    global_key,
    local_key,
    numeral,
    form=None,
    figbass=None,
    relativeroot=None,
    no_inversion=True,
):
    if numeral not in ["Ger", "It", "Fr"]:
        local_key_name = get_local_chord_from_key_and_numeral(global_key, local_key)
        if relativeroot is not None:
            relativeroot_name = get_local_chord_from_key_and_numeral(
                local_key_name, relativeroot
            )
            chord_name = get_local_chord_from_key_and_numeral(
                relativeroot_name, numeral
            )
        else:
            chord_name = get_local_chord_from_key_and_numeral(local_key_name, numeral)

        if form == "%":
            if figbass not in ["65", "43", "42", "2"]:
                figbass = "7"

        if form is not None:
            root = root_upper(chord_name)
            chord_name = root + form
        else:
            if chord_name == chord_name.lower():
                root = root_upper(chord_name)
                chord_name = root + "m"

        if no_inversion:
            if figbass in ["9", "7", "65", "43", "42", "2"]:
                figbass = "7"
            if figbass in ["6", "64"]:
                figbass = None

        if figbass is not None:
            chord_name += figbass
    else:
        local_key_name = get_local_chord_from_key_and_numeral(global_key, local_key)
        if relativeroot is not None:
            local_key_name = get_local_chord_from_key_and_numeral(
                local_key_name, relativeroot
            )

        local_key_name = root_upper(local_key_name)
        chord_name = local_key_name + numeral + "6"

    return chord_name


def get_local_chord_from_key_and_numeral(key, numeral):
    chromatic_scale_with_sharp = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]
    chromatic_scale_with_flat = [
        "C",
        "Db",
        "D",
        "Eb",
        "E",
        "F",
        "Gb",
        "G",
        "Ab",
        "A",
        "Bb",
        "B",
    ]

    chromatic_scale_with_enharmonic_sharp = [
        "B#",
        "C#",
        "D",
        "D#",
        "E",
        "E#",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]
    chromatic_scale_with_enharmonic_flat = [
        "C",
        "Db",
        "D",
        "Eb",
        "Fb",
        "F",
        "Gb",
        "G",
        "Ab",
        "A",
        "Bb",
        "Cb",
    ]

    major_scale = [0, 2, 4, 5, 7, 9, 11]
    minor_scale = [0, 2, 3, 5, 7, 8, 10]

    minor_mode = key == key.lower()
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
    chord_position = (key_position + diatonic_scale[scale_degree - 1]) % 12
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


def parse_numeral(numeral):
    accidental = numeral.count("#") - numeral.count("b")

    numeral = numeral.replace("#", "")
    numeral = numeral.replace("b", "")
    major_quality = numeral == numeral.upper()

    numeral = numeral.upper()
    scale_degree_list = ["I", "II", "III", "IV", "V", "VI", "VII"]
    scale_degree = scale_degree_list.index(numeral) + 1

    return scale_degree, accidental, major_quality


def root_upper(root):
    return root[0].upper() + root[1:]


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
