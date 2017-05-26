from os.path import dirname, join
import pycrfsuite
from transformer import Transformer
from underthesea.util.singleton import Singleton


@Singleton
class ChunkingCRFModel:
    def __init__(self):
        self.model = pycrfsuite.Tagger()
        filepath = join(dirname(__file__), "chunking_crf_v1.model")
        self.model.open(filepath)

    def predict(self, sentence, format=None):
        x = Transformer.transform(sentence)
        tags = self.model.tag(x)
        tokenized_sentence = u''
        output = [(token[0], token[1], tag) for token, tag in zip(sentence, tags)]
        return output
