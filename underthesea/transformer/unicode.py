class UnicodeTransformer:
    def __init__(self):
        pass

    def transform(self, text):
        """
        :param text
        :type text: unicode

        """
        if type(text) is not unicode:
            text = unicode(text, 'utf-8')
        return text

