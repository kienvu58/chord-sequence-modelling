import numpy as np


def perplexity(log_score_list):
    r"""
    Calculates perplexity of a corpus.
    The corpus contain m sentences, assumably independent.
        log_score_list: list of tuples, each tuple contains the log-probability
            of the sentences and the number of words in the sentence.
    Let p(s_i) is the probability of sentence ith, N is the total number
    of words in the corpus C.
        perplexity(C) = [\prod_{i=1}^m p(s_i)]^(-1/N)
                      = \exp{ \log [\prod_{i=1}^m p(s_i)]^(-1/N) }
                      = \exp{ -1/N \sum_{i=1}^m \log p(s_i) }
    """
    log_scores, ns = zip(*log_score_list)
    H = np.sum(log_scores)

    N = np.sum(ns)
    H /= N

    pp = np.exp(-H)
    return pp


def evaluate(model, corpus):
    log_score_list = []
    for s in corpus:
        log_score = model.log_score(s)
        n = len(s) + 1  # 1 for ending symbol
        log_score_list.append((log_score, n))

    pp = perplexity(log_score_list)
    return pp
