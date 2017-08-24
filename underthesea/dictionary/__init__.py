import pickle
from os.path import join, dirname

from underthesea.util.singleton import Singleton

@Singleton
class Dictionary:
    def __init__(self):
        """ contains 13k words in this version
        """
        filepath = join(dirname(__file__), "dictionary.data")
        self.words = pickle.load(open(filepath, "rb"))

    def lookup(self, text):
        if text in self.words:
            return self.words[text]
        return None
