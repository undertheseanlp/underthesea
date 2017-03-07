class UnicodeTransformer:
    def __init__(self):
        pass

    def transform(self, text):
        """
        :param unicode|str text: input text
        :type text: unicode|str

        :rtype unicode
        :return transformed text
        """
        if type(text) is not unicode:
            text = unicode(text, 'utf-8')
        return text

