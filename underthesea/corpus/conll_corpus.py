from underthesea.corpus import Corpus


class CONLLCorpus(Corpus):
    def __init__(self, train, dev=None, test=None, name: str = 'corpus'):
        self.name: str = name
        self._train = train
        self._dev = dev
        self._test = test

    @property
    def train(self):
        return self._train

    @property
    def dev(self):
        return self._dev

    @property
    def test(self):
        return self._test
