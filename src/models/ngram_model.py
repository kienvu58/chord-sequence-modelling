from .model_interface import ModelI
from nltk.lm.models import (
    Lidstone,
    Laplace,
    WittenBellInterpolated,
    KneserNeyInterpolated
)
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import ngrams, pad_sequence
import numpy as np


class NgramModel(ModelI):
    def __init__(self, hparams):
        self.order = hparams["order"]
        self.counter = hparams["counter"]
        self.vocab = hparams["vocab"]
        model_class_name = hparams["model_class"]
        kwargs = hparams["kwargs"]
        if model_class_name in ["Lidstone", "Laplace"]:
            model_class = globals()[model_class_name]
            self.model = model_class(
                self.order, self.vocab, self.counter, **kwargs)
        elif model_class_name in ["WittenBellInterpolated", "KneserNeyInterpolated"]:
            model_class = globals()[model_class_name]
            self.model = model_class(self.order, **kwargs)
        else:
            raise ValueError("Unsupported model type", model_class_name)

    def log_score(self, progression):
        progression = list(pad_sequence(progression, self.order, pad_left=True,
                                        left_pad_symbol="<s>", pad_right=True, right_pad_symbol="</s>"))
        contexts = list(ngrams(progression, self.order-1))[:-1]
        words = progression[self.order-1:]

        total_log_score = 0
        for word, context in zip(words, contexts):
            score = self.model.score(word, context)
            log_score = np.log(score)
            total_log_score += log_score

        return total_log_score

    def fit(self, dataset):
        train, vocab = padded_everygram_pipeline(self.order, dataset)
        if self.vocab is None:
            self.vocab = vocab
        self.model.fit(train, self.vocab)

    def predict(self, context):
        context = (
            context[-self.order + 1:]
            if len(context) >= self.order
            else context
        )
        samples = self.model.context_counts(self.model.vocab.lookup(context))
        while context and not samples:
            context = context[1:] if len(context) > 1 else []
            samples = self.model.context_counts(self.model.vocab.lookup(context))

        samples = sorted(samples)
        scores = [self.model.score(w, context) for w in samples]
        next_index = np.argmax(scores)
        return samples[next_index]

    def generate(self, min_length=1, max_length=100):
        finished = False
        progression = ["<s>"]
        while not finished:
            next_chord = self.model.generate(text_seed=progression)
            if next_chord not in ["</s>", "<s>"]:
                progression.append(next_chord)
            finished = next_chord == "</s>" and len(
                progression) > min_length or len(progression) > max_length

        return progression[1:]
