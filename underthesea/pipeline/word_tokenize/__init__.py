from os.path import dirname, join

import joblib
from underthesea_core import CRFTagger

from .regex_tokenize import tokenize

word_tokenize_model = None


def word_tokenize(sentence, format=None, use_token_normalize=True, fixed_words=None):
    """
    Vietnamese word segmentation

    Args:
        sentence (str): raw sentence
        format (str, optional): format option.
            Defaults to None.
            use format=`text` for text format
        use_token_normalize (bool): True if use token_normalize
        fixed_words (list): list of fixed words

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
    if fixed_words is None:
        fixed_words = []
    global word_tokenize_model
    tokens = tokenize(sentence, use_token_normalize=use_token_normalize, fixed_words=fixed_words)
    features = [[token] for token in tokens]
    if word_tokenize_model is None:
        word_tokenize_model = CRFTagger()
        wd = dirname(__file__)
        model_dir = join(wd, "models", "ws_crf_vlsp2013_20230727")
        word_tokenize_model.load(join(model_dir, "models.bin"))
        features_config = joblib.load(join(model_dir, "features.bin"))
        dictionary = joblib.load(join(model_dir, "dictionary.bin"))
        word_tokenize_model.set_featurizer(features_config, dictionary)
    tags = word_tokenize_model.predict(features)

    output = []
    num_words = 0
    for tag, token in zip(tags, tokens):
        if tag == "I-W" and num_words > 0:
            output[-1] = output[-1] + " " + token
        else:
            output.append(token)
        num_words += 1
    if format == "text":
        output = " ".join([item.replace(" ", "_") for item in output])
    return output
