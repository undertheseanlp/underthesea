from underthesea.feature_engineering.feature import sent2features


class Transformer:
    def __init__(self):
        pass

    @staticmethod
    def transform(sentence):
        template = [
            "T[0].lower", "T[-1].lower", "T[1].lower",
            "T[0].istitle", "T[-1].istitle", "T[1].istitle",
            "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",  # unigram
            "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",  # bigram
            "T[-1][1]", "T[-2][1]", "T[-3][1]",  # dynamic feature
            "T[-3,-2][1]", "T[-2,-1][1]",
            "T[-3,-1][1]"
        ]
        sentence = [(token, "A") for token in sentence]
        return sent2features(sentence, template)

    @staticmethod
    def extract_features(sentence, template):
        return sent2features(sentence, template)


