from corpus import Corpus
from os.path import join
from os import listdir


class PlainTextCorpus(Corpus):
    """class for handling plain text corpus
    """

    def __init__(self):
        self.documents = None

    def load(self, folder):
        """load plaintext folder to documents

        :param folder path to directory
        :type folder: string
        """
        files = listdir(folder)
        files = [join(folder, f) for f in files]
        self.documents = [open(f, "r").read() for f in files]
        pass

    def save(self, folder):
        pass
