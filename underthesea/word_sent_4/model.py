from os.path import dirname, join
import pycrfsuite
from underthesea.util.singleton import Singleton
from underthesea.word_sent_4.transformer import Transformer


@Singleton
class CRFModel:
    def __init__(self):
        self.model = pycrfsuite.Tagger()
        filepath = join(dirname(__file__), "wordsent_crf_4_20171009.model")
        self.model.open(filepath)

    def predict(self, sentence, format=None):
        x = Transformer.transform(sentence)
        tags = self.model.tag(x)
        return list(zip(sentence, tags))
