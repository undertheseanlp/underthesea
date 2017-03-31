from os.path import dirname, join

import pycrfsuite

from underthesea.word_sent.transformer import Transformer


class CRFModel:
    def __init__(self):
        """
        load trained data from crf-model
        """
        self.model = pycrfsuite.Tagger()
        filepath = join(dirname(__file__), "crf-model")
        self.model.open(filepath)
        filepath = join(dirname(__file__), "punctuation.txt")
        self.punctuation = open(filepath, "r").read().split("\n")

    def predict(self, sentence):
        """

        :param unicode|str sentence: raw sentence
        :return: segmented sentence
        :rtype: unicode|str
        """
        sentence = Transformer.transform(sentence)
        tags = self.model.tag(sentence)
        tokenized_sentence = u''
        for tag, word in zip(tags, sentence):
            word = word[0]
            if tag == "I_W":
                tokenized_sentence = tokenized_sentence + u"_" + word
            else:
                tokenized_sentence = tokenized_sentence + word
            tokenized_sentence += " "
        format_sentence = ''
        for word in tokenized_sentence.split("_"):
            if word not in self.punctuation:
                format_sentence += word[: -1] + "_"
        tokenized_sentence = format_sentence[:-1]
        return tokenized_sentence
