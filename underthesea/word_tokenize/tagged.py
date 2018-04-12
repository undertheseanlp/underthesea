from .tagged_feature import word2features
import re

class TaggedTransformer:
    def __init__(self, template=None):
        self.template = template
        self.template_function = [self.token_function(token_syntax) for token_syntax in template]

    def token_function(self, token_syntax):
        matched = re.match(
            "T\[(?P<index1>\-?\d+)(\,(?P<index2>\-?\d+))?\](\[(?P<column>.*)\])?(\.(?P<function>.*))?",
            token_syntax)
        column = matched.group("column")
        column = int(column) if column else 0
        index1 = int(matched.group("index1"))
        index2 = matched.group("index2")
        index2 = int(index2) if index2 else None
        func = matched.group("function")
        return column, index1, index2, func, token_syntax

    def transform(self, sentences):
        X = [self.sentence2features(s) for s in sentences]
        y = [self.sentence2labels(s) for s in sentences]
        return X, y

    def sentence2features(self, s):
        return [word2features(s, i, self.template_function) for i in range(len(s))]

    def sentence2labels(self, s):
        return [row[-1] for row in s]
