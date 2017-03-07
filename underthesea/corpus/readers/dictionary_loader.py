from os.path import join, dirname
from underthesea.transformer.unicode import UnicodeTransformer


class DictionaryLoader:
    def __init__(self, filepath):
        """load words from Ho Ngoc Duc's dictionary

        :param str filepath: filename of dictionary data
        :type filepath: str
        """
        data_folder = join(dirname(dirname(__file__)), "data")
        data_file = join(data_folder, filepath)
        self.data_file = data_file
        self.words_data = None

    @property
    def words(self):
        if not self.words_data:
            words = open(self.data_file, "r").read().splitlines()
            unicode_transformer = UnicodeTransformer()
            words = [unicode_transformer.transform(word) for word in words]
            self.words_data = words
        return self.words_data
