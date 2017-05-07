from os.path import dirname
from os.path import join

from features.feature_2 import word2features
from corpus import TaggedCorpus


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2features_2(sent, template):
    return [word2features(sent, i, template) for i in range(len(sent))]



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

    @staticmethod
    def extract_features_2(sentence, template):
        return sent2features_2(sentence, template)


    def format_word(self, sentence):
        path = join(dirname(dirname(dirname(__file__))), "pipelines", "logs", "punctuation.txt")
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
        corpus = TaggedCorpus()
        file_path = join(dirname(dirname(dirname(__file__))), "data", "ud", "vi_ud.conllu")
        corpus.load(file_path)
        sentences = corpus.sents()
        return sentences


def sent2labels(sent):
    return [label for token, label in sent]
