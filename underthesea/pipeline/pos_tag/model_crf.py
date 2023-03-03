from os.path import join, dirname
from underthesea.util.singleton import Singleton
import pycrfsuite
from .tagged_feature import word2features


# [Deprecating] This file will be deprecated in the version 6.5.0

@Singleton
class CRFPOSTagPredictor:
    def __init__(self):
        self.model = pycrfsuite.Tagger()
        filepath = join(dirname(__file__), "pos_crf_2017_10_11.bin")
        self.model.open(filepath)

        template = [
            "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower",
            "T[2].lower",
            "T[0].istitle", "T[-1].istitle", "T[1].istitle",
            # word unigram and bigram
            "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
            "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
            # pos unigram and bigram
            "T[-3][1]", "T[-2][1]", "T[-1][1]",
            "T[-3,-2][1]", "T[-2,-1][1]",
        ]
        self.transformer = TaggedTransformer(template)

    def predict(self, sentence, format=None):
        tokens = [(token, "X") for token in sentence]
        x = self.transformer.transform([tokens])[0][0]
        tags = self.model.tag(x)
        return list(zip(sentence, tags))


class TaggedTransformer:
    def __init__(self, template=None):
        self.template = template

    def transform(self, sentences):
        X = [self.sentence2features(s) for s in sentences]
        y = [self.sentence2labels(s) for s in sentences]
        return X, y

    def sentence2features(self, s):
        return [word2features(s, i, self.template) for i in range(len(s))]

    def sentence2labels(self, s):
        return [row[-1] for row in s]
