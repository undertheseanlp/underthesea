from corpus import Corpus


class PlainTextCorpus(Corpus):
    """corpus for handling plain text
    """
    def __init__(self):
        self.documents = None

    def load(self, folder):
        pass

    def save(self, folder):
        pass
