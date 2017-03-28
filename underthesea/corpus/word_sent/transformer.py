from os.path import dirname
from os.path import join

from underthesea.corpus import PlainTextCorpus
from underthesea.corpus.word_sent.feature_selection.feature_2 import word2features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


class Transformer:
    def __init__(self):
        pass

    @staticmethod
    def transform(sentence):
        sentence = [(token,) for token in sentence.split()]
        return sent2features(sentence)

    @staticmethod
    def extract_features(sentence):
        return sent2features(sentence)


def sent2labels(sent):
    return [label for token, label in sent]
