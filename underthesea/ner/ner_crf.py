import pycrfsuite
from os.path import dirname, join

from underthesea.feature_engineering.feature import sent2features
from underthesea.util.singleton import Singleton


@Singleton
class NERCRFModel:
    def __init__(self):
        self.model = pycrfsuite.Tagger()
        filepath = join(dirname(__file__), "ner_crf_20171006_template_2.model")
        # filepath = join(dirname(__file__), "ner_crf_20171005.model")
        self.model.open(filepath)

    def predict(self, text, format=None):
        x = Transformer.transform(text)
        tags = self.model.tag(x)
        output = [(token[0], token[1], token[2], tag) for token, tag in
                  zip(text, tags)]
        return output


class Transformer:
    def __init__(self):
        pass

    @staticmethod
    def transform(sentence):
        template = [
            "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower", "T[2].lower",
            "T[0].istitle", "T[-1].istitle", "T[1].istitle",
            # word unigram and bigram
            "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
            "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
            # pos unigram and bigram
            "T[-2][1]", "T[-1][1]", "T[0][1]", "T[1][1]", "T[2][1]",
            "T[-2,-1][1]", "T[-1,0][1]", "T[0,1][1]", "T[1,2][1]",
            # chunk unigram and bigram
            "T[-2][2]", "T[-1][2]", "T[0][2]", "T[1][2]", "T[2][2]",
            "T[-2,-1][2]", "T[-1,0][2]", "T[0,1][2]", "T[1,2][2]",
            # ner
            "T[-3][3]", "T[-2][3]", "T[-1][3]",
        ]
        template2 = [
            "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower",
            "T[2].lower",
            "T[0].istitle", "T[-1].istitle", "T[1].istitle", "T[-2].istitle",
            "T[2].istitle",
            # word unigram and bigram
            "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
            "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
            # pos unigram and bigram
            "T[-2][1]", "T[-1][1]", "T[0][1]", "T[1][1]", "T[2][1]",
            "T[-2,-1][1]", "T[-1,0][1]", "T[0,1][1]", "T[1,2][1]",
            # ner
            "T[-3][3]", "T[-2][3]", "T[-1][3]",
        ]
        sentence = [(token[0], token[1], token[2], "X") for token in sentence]
        return sent2features(sentence, template2)

    @staticmethod
    def extract_features(sentence, template):
        return sent2features(sentence, template)
