from os.path import dirname
from os.path import join

from underthesea.pos_tag.feature import word2features


def sent2features(sent, template):
    return [word2features(sent, i, template) for i in range(len(sent))]


class Transformer:
    def __init__(self):
        pass

    @staticmethod
    def transform(sentence):
        template = [
            "T[0].lower", "T[-1].lower", "T[1].lower",
            "T[0].istitle", "T[-1].istitle", "T[1].istitle",
            "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",  # unigram
            "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",  # bigram
            "T[-1][1]", "T[-2][1]", "T[-3][1]",  # dynamic feature
            "T[-3,-2][1]", "T[-2,-1][1]",
            "T[-3,-1][1]"
        ]
        sentence = [(token, "A") for token in sentence]
        return sent2features(sentence, template)

    @staticmethod
    def extract_features(sentence, template):
        return sent2features(sentence, template)

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


def sent2labels(sent):
    return [label for token, label in sent]
