import pickle
from os.path import join, dirname


class UVD:
    """
    Underthesea Vietnamese Dictionary
    """
    version = '0.0.1'
    date = '2022/03/05'

    def __init__(self):
        """ contains 13k words in this version
        """
        filepath = join(dirname(__file__), "dictionary.data")
        self.words = []
        with open(filepath, "rb") as f:
            self.words = pickle.load(f)


class DictionaryStat:
    @staticmethod
    def stat(dictionary):
        print('Words:', len(dictionary.words))
