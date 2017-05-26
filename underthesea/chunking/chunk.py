from underthesea import pos_tag
from underthesea.chunking.model import ChunkingCRFModel


def chunk(sentence, format=None):
    """
    chunk a sentence to phrases 
    
    :param unicode sentence: raw sentence
    :return: list of tuple with word, pos tag, chunking tag 
    :rtype: list 
    """
    sentence = pos_tag(sentence)
    crf_model = ChunkingCRFModel.Instance()
    result = crf_model.predict(sentence, format)
    return result

