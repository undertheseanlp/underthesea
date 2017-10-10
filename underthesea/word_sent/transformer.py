from underthesea.word_sent.feature import word2features


def sent2features(sent):
    """

    :param unicode|str sent: sentence has been converted to column format
    :return: words of sentence added feature
    :rtype: list
    """
    return [word2features(sent, i) for i in range(len(sent))]


class Transformer:
    def __init__(self):
        pass

    @staticmethod
    def transform(text):
        """
        :param unicode text: raw sentence
        :return: words of sentence added feature
        :rtype: list
        """
        sentence = [(token,) for token in text.split()]
        return sent2features(sentence)

