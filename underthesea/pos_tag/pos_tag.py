from underthesea.pos_tag.predict import predict


def pos_tag(sentence, text=False):
    """

    :param sentence: raw sentence
    :param option: bool
    :return: pos tagged sentence
    """
    return predict(sentence)
