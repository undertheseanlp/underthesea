# -*- coding: utf-8 -*-


# References: https://github.com/flairNLP/flair/blob/master/flair/data.py#L1049
class Corpus:
    """Corpus is fundamental resource of NLP
    """

    def __init__(self):
        pass

    def load(self, folder):
        pass

    def save(self, folder):
        pass

    @property
    def train(self):
        return self._train

    @property
    def dev(self):
        return self._dev

    @property
    def test(self):
        return self._test
