import sys

if sys.version_info >= (3, 0):
    unicode = str


class UnicodeTransformer:
    def __init__(self):
        pass

    def transform(self, text):
        """
        :param unicode|str text: input text
        :type text: unicode|str

        :return: transformed text
        :rtype: unicode
        """
        if type(text) is not unicode:
            text = unicode(text, 'utf-8')
        return text
