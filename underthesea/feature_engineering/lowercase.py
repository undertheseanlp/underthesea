class LowercaseTransformer:
    def __init__(self):
        pass

    def transform(self, text):
        """
        :param unicode|str text: input text
        :type text: unicode|str

        :return: transformed text
        :rtype: unicode|str
        """
        return text.lower()
