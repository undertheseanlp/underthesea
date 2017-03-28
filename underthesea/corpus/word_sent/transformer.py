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

    def load_train_sents(self):
        corpus = PlainTextCorpus()
        file_path = join(dirname(dirname(dirname(__file__))), "data", "corpus_2", "train", "input")
        corpus.load(file_path)
        sentences = []
        for document in corpus.documents:
            for sentence in document.sentences:
                if sentence != "":
                    sentences.append(sentence)
        return sentences


def sent2labels(sent):
    return [label for token, label in sent]
