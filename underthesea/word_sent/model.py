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

    def predict(self, sentence, format=None):
        """

        :param unicode|str sentence: raw sentence
        :return: segmented sentence
        :rtype: unicode|str
        """
        tokens = Transformer.transform(sentence)
        tags = self.model.tag(tokens)
        tokens = [item[0] for item in tokens]
        output = []
        for tag, token in zip(tags, tokens):
            if tag == "I_W":
               output[-1] = output[-1] + u" " + token
            else:
                output.append(token)
        if format == "text":
            output = u" ".join([item.replace(" ", "_") for item in output])
        return output
