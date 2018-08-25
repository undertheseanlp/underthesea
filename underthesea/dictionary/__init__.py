import pickle
from os.path import join, dirname

from underthesea.util.singleton import Singleton


@Singleton
class Dictionary:
    def __init__(self):
        """ contains 13k words in this version
        """
        filepath = join(dirname(__file__), "dictionary.data")
        with open(filepath, "rb") as f:
            self.words = pickle.load(f)

    def lookup(self, text):
        if text in self.words:
            return self.words[text]
        return None
