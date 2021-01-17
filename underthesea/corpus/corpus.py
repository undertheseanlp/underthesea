# -*- coding: utf-8 -*-


# References: https://github.com/flairNLP/flair/blob/master/flair/data.py#L1049
class Corpus:
    """Corpus is fundamental resource of NLP
    """

    def __init__(self):
        self._train = None
        self._dev = None
        self._test = None

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

    def downsample(self, percentage: float = 0.1, downsample_train=True, downsample_dev=True, downsample_test=True):
        """ TODO: implement this
        Ref: Flair

        :param percentage:
        :param downsample_train:
        :param downsample_dev:
        :param downsample_test:
        :return:
        """
        if downsample_train:
            self._train = self._downsample_to_proportion(self.train, percentage)

        if downsample_dev:
            self._dev = self._downsample_to_proportion(self.dev, percentage)

        if downsample_test:
            self._test = self._downsample_to_proportion(self.test, percentage)

        return self
