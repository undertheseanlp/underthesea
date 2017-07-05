from os.path import dirname, join
import pycrfsuite
import sys
from underthesea.pos_tag.transformer import Transformer
from underthesea.util.singleton import Singleton


@Singleton
class CRFModel:
    def __init__(self):
        self.model = pycrfsuite.Tagger()
        filepath = join(dirname(__file__), "postag_crf_v1.model")
        self.model.open(filepath)

    def predict(self, sentence, format=None):
        x = Transformer.transform(sentence)
        tags = self.model.tag(x)
        return list(zip(sentence, tags))
