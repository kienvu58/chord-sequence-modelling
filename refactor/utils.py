import itertools

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN


def generate_vocab():
    note_list = ["A", "B", "C", "D", "E", "F", "G"]
    accidental_list = ["", "b", "#"]
    chord_type_list = ["", "m", "+", "o", "7", "m7", "M7", "o7", "%7", "+7", "It6", "Ger6", "Fr6"]

    vocab = Vocabulary()

    for chord in itertools.product(note_list, accidental_list, chord_type_list):
        vocab.add_token_to_namespace("".join(chord))

    vocab.add_token_to_namespace(START_SYMBOL)
    vocab.add_token_to_namespace(END_SYMBOL)

    vocab.save_to_files("data/vocabulary")

generate_vocab()

vocab = Vocabulary.from_files("data/vocabulary")

print(vocab.get_token_to_index_vocabulary())

