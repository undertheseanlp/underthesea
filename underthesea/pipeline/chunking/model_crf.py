from os.path import join, dirname
from underthesea.util.singleton import Singleton
import pycrfsuite
import sys

if sys.version_info >= (3, 0):
    from .tagged_feature import word2features
else:
    from tagged_feature import word2features


@Singleton
class CRFChunkingPredictor:
    def __init__(self):
        self.model = pycrfsuite.Tagger()
        filepath = join(dirname(__file__), "chunk_crf_2017_10_12.bin")
        self.model.open(filepath)

        template = [
            "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower",
            "T[2].lower",
            "T[0].istitle", "T[-1].istitle", "T[1].istitle",
            # word unigram and bigram
            "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
            "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
            # pos unigram and bigram
            "T[-2][1]", "T[-1][1]", "T[0][1]", "T[1][1]", "T[2][1]",
            "T[-2,-1][1]", "T[-1,0][1]", "T[0,1][1]", "T[1,2][1]",
            # chunk
            "T[-3][2]", "T[-2][2]", "T[-1][2]",
        ]
        self.transformer = TaggedTransformer(template)

    def predict(self, sentence, format=None):
        tokens = [(tokens[0], tokens[1], "X") for tokens in sentence]
        x = self.transformer.transform([tokens])[0][0]
        tags = self.model.tag(x)
        output = [(tokens[0], tokens[1], tag) for tokens, tag in
                  zip(sentence, tags)]
        return output


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
