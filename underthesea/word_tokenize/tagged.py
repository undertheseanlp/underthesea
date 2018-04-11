from .tagged_feature import word2features


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
