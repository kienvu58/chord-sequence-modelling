import os
import subprocess

if os.name == "nt":
    lilypond_path = "C:/Program Files (x86)/LilyPond/usr/bin/lilypond.exe"
else:
    lilypond_path = "lilypond"

def convert_to_note_set(chord):
    note_name_list = ["c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b"]
    note_set_index = [int(note) for note in chord.split("_")]
    note_set = [note_name_list[int(note)] for note in chord.split("_")]

    if len(note_set) < 4:
        cello = note_set[0]
        viola = note_set[1]
        violin_two = note_set[2]
        violin_one = note_set[2]
    else:
        cello = note_set[0]
        viola = note_set[1]
        violin_two = note_set[2]
        violin_one = note_set[3]

    if note_set_index[1] < note_set_index[0]:
        viola += "'"
        violin_one += "'"
        violin_two += "'"

    # cello += ","
    violin_two += "'"
    violin_one += "'"

    return (cello, viola, violin_two, violin_one)


def convert_progression_to_lilypond_format(progression):
    if not isinstance(progression, list):
        progression = progression.split(" ")
    note_set_list = []
    for chord in progression:
        note_set = convert_to_note_set(chord)
        note_set_list.append(note_set)

    cello, viola, violin_two, violin_one = list(zip(*note_set_list))
    voice = lambda l: " ".join(l)
    return voice(cello), voice(viola), voice(violin_two), voice(violin_one) 

def generate_bass_notes(progression, title, path):
    template = r"""
    \version "2.18.2"

    \header {
        title = "<title>"
    }

    global= {
        \time 4/4
        \key c \major
    }

    cello = \new Voice \absolute {
        \clef bass
        <cello>
        \bar "|."
    }

    \score {
        \new StaffGroup <<
            \new Staff << \global \cello >>
        >>
        \layout { }
        \midi { }
    }
    """

    template = template.replace("<title>", title)
    cello, _, _, _ = convert_progression_to_lilypond_format(progression)
    template = template.replace("<cello>", cello)

    if not os.path.isdir(path):
        os.makedirs(path)

    score_fn = "{}.ly".format(title)
    score_path = os.path.join(path, score_fn) 
    with open(score_path, "w") as f:
        f.write(template)

    os.chdir(path)
    subprocess.call([lilypond_path, score_fn])

def generate_score_and_audio(progression, title, path):
    template = r"""
    \version "2.18.2"

    \header {
        title = "<title>"
    }

    global= {
        \time 4/4
        \key c \major
    }

    violinOne = \new Voice \absolute {
        <violinOne>
        \bar "|."
    }

    violinTwo = \new Voice \absolute {
        <violinTwo>
        \bar "|."
    }

    viola = \new Voice \absolute {
        \clef alto
        <viola>
        \bar "|."
    }

    cello = \new Voice \absolute {
        \clef bass
        <cello>
        \bar "|."
    }

    \score {
        \new StaffGroup <<
            \new Staff << \global \violinOne >>
            \new Staff << \global \violinTwo >>
            \new Staff << \global \viola >>
            \new Staff << \global \cello >>
        >>
        \layout { }
        \midi { }
    }
    """

    template = template.replace("<title>", title)
    cello, viola, violin_two, violin_one = convert_progression_to_lilypond_format(progression)
    template = template.replace("<cello>", cello)
    template = template.replace("<viola>", viola)
    template = template.replace("<violinTwo>", violin_two)
    template = template.replace("<violinOne>", violin_one)

    if not os.path.isdir(path):
        os.makedirs(path)

    score_fn = "{}.ly".format(title)
    score_path = os.path.join(path, score_fn) 
    with open(score_path, "w") as f:
        f.write(template)

    owd = os.getcwd()
    os.chdir(path)
    subprocess.call([lilypond_path, score_fn])
    os.chdir(owd)