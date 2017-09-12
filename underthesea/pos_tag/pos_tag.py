from underthesea import word_sent
from underthesea.pos_tag.model import CRFModel


def pos_tag(sentence, format=None):
    """
    part of speech tagging
    
    :param unicode|str sentence: raw sentence
    :return: tagged sentence 
    :rtype: list 
    """
    sentence = word_sent(sentence)
    crf_model = CRFModel.Instance()
    result = crf_model.predict(sentence, format)
    return result

