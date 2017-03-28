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

    def format_word(self, sentence):
        path = join(dirname(__file__), "punctuation.txt")
        punctuations = open(path, "r").read().split("\n")
        for punctuation in punctuations:
            punctuation = unicode(punctuations)
        words = []
        for word in sentence.split(" "):
            if "_" in word:
                tokens = []
                word = word.replace("_", " ")
                for token in word.split(" "):
                    if token != "":
                        tokens.append(token)

                for i in range(tokens.__len__()):
                    if i != 0:
                        tokens[i] += "\tI_W"
                    else:
                        tokens[i] += "\tB_W"
                    words.append(tokens[i])
            elif word in punctuations:
                words.append(word + "\tO")
            else:
                words.append(word + "\tB_W")
        return words

    def list_to_tuple(self, sentences):
        word_tuple = []
        for i in sentences:
            arr = i.split('\t')
            word_tuple.append((arr[0], arr[1]))
        return word_tuple

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
