from underthesea.feature_engineering.feature import sent2features
from underthesea.word_sent.tokenize import tokenize


class Transformer:
    def __init__(self):
        pass

    @staticmethod
    def transform(sentence):
        template = [
            "T[-1].isdigit", "T[0].isdigit", "T[1].isdigit",
            "T[-1].istitle", "T[0].istitle", "T[1].istitle",
            "T[0,1].istitle", "T[0,2].istitle",
            "T[-1].is_in_dict", "T[0].is_in_dict", "T[1].is_in_dict",
            "T[0,1].is_in_dict", "T[0,2].is_in_dict",
            # word unigram and bigram and trigram
            "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
            "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
            "T[-2,0]", "T[-1,1]", "T[0,2]",
            # BI tag
            "T[-2][1]", "T[-1][1]"
        ]
        sentence = [(token, "X") for token in sentence]
        return sent2features(sentence, template)

    @staticmethod
    def extract_features(sentence, template):
        return sent2features(sentence, template)


