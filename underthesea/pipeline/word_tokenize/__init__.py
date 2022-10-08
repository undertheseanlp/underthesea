# -*- coding: utf-8 -*-
from .regex_tokenize import tokenize
from .model import CRFModel


def word_tokenize(sentence, format=None, use_token_normalize=True):
    """
    Vietnamese word segmentation

    Args:
        sentence (str): raw sentence
        format (str, optional): format option.
            Defaults to None.
            use format=`text` for text format
        use_token_normalize (bool): True if use token_normalize

    Returns:
        :obj:`list` of :obj:`str`:
            word tokens

    Examples:

        >>> # -*- coding: utf-8 -*-
        >>> from underthesea import word_tokenize
        >>> sentence = "Bác sĩ bây giờ có thể thản nhiên báo tin bệnh nhân bị ung thư"

        >>> word_tokenize(sentence)
        ["Bác sĩ", "bây giờ", "có thể", "thản nhiên", "báo tin", "bệnh nhân", "bị", "ung thư"]

        >>> word_tokenize(sentence, format="text")
        "Bác_sĩ bây_giờ có_thể thản_nhiên báo_tin bệnh_nhân bị ung_thư"
    """
    tokens = tokenize(sentence, use_token_normalize=use_token_normalize)
    crf_model = CRFModel.instance()
    output = crf_model.predict(tokens, format)
    tokens = [token[0] for token in output]
    tags = [token[1] for token in output]
    output = []
    num_words = 0
    for tag, token in zip(tags, tokens):
        if tag == "I-W" and num_words > 0:
            output[-1] = output[-1] + u" " + token
        else:
            output.append(token)
        num_words += 1
    if format == "text":
        output = u" ".join([item.replace(" ", "_") for item in output])
    return output
