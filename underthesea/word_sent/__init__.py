from .regex_tokenize import tokenize
from .model_crf import CRFModel


def word_sent(sentence, format=None):
    """
    word segmentation

    Parameters
    ==========

    sentence: raw sentence

    Returns
    =======
    tokens: list of text
        tagged sentence
    """
    sentence = tokenize(sentence).split()
    crf_model = CRFModel.Instance()
    output = crf_model.predict(sentence, format)
    tokens = [token[0] for token in output]
    tags = [token[1] for token in output]
    output = []
    for tag, token in zip(tags, tokens):
        if tag == "IW":
            output[-1] = output[-1] + u" " + token
        else:
            output.append(token)
    if format == "text":
        output = u" ".join([item.replace(" ", "_") for item in output])
    return output
