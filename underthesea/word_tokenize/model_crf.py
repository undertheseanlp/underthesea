from os.path import dirname, join
import pycrfsuite
import sys
from underthesea.util.singleton import Singleton
from .transformer import CustomTransformer


@Singleton
class CRFModel:
    def __init__(self):
        self.model = pycrfsuite.Tagger()
        filepath = join(dirname(__file__), "model_9.bin")
        self.model.open(filepath)
        template = [
            "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower",
            "T[2].lower",
            "T[-1].isdigit", "T[0].isdigit", "T[1].isdigit",

            "T[-1].istitle", "T[0].istitle", "T[1].istitle",

            "T[0,1].istitle", "T[0,2].istitle",

            "T[-2].is_in_dict", "T[-1].is_in_dict", "T[0].is_in_dict",
            "T[1].is_in_dict", "T[2].is_in_dict",

            "T[-2,-1].is_in_dict", "T[-1,0].is_in_dict", "T[0,1].is_in_dict",
            "T[1,2].is_in_dict",

            "T[-2,0].is_in_dict", "T[-1,1].is_in_dict", "T[0,2].is_in_dict",

            # word unigram and bigram and trigram
            "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
            "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
            "T[-2,0]", "T[-1,1]", "T[0,2]",
            # BI tag
            "T[-2][1]", "T[-1][1]"
        ]
        self.transformer = CustomTransformer(template)

    def predict(self, sentence, format=None):

        tokens = [(token, "X") for token in sentence]
        x = self.transformer.transform([tokens])[0][0]
        tags = self.model.tag(x)
        return list(zip(sentence, tags))
