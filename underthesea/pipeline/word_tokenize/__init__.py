from os.path import dirname, join

from .regex_tokenize import tokenize

word_tokenize_model = None


def word_tokenize(sentence, format=None, use_token_normalize=True, fixed_words=None,
                  restore_diacritics=False):
    """
    Vietnamese word segmentation

    Args:
        sentence (str): raw sentence
        format (str, optional): format option.
            Defaults to None.
            use format=`text` for text format
        use_token_normalize (bool): True if use token_normalize
        fixed_words (list): list of fixed words
        restore_diacritics (bool): True to restore Vietnamese diacritics
            before segmentation, for text written without them
            (e.g. "lap trinh vien"). Defaults to False.

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

        >>> word_tokenize("lap trinh vien tri tue nhan tao", restore_diacritics=True)
        ["lập trình viên", "trí tuệ", "nhân tạo"]
    """
    if fixed_words is None:
        fixed_words = []
    global word_tokenize_model
    if restore_diacritics:
        # Lazy import: the diacritics model is only loaded when requested
        from underthesea.pipeline.restore_diacritics import restore_diacritics as _restore
        sentence = _restore(sentence)
    tokens = tokenize(sentence, use_token_normalize=use_token_normalize, fixed_words=fixed_words)
    features = [[token] for token in tokens]
    if word_tokenize_model is None:
        # Lazy import to avoid loading underthesea_core at module load time
        from underthesea.models.fast_crf_sequence_tagger import FastCRFSequenceTagger
        word_tokenize_model = FastCRFSequenceTagger()
        wd = dirname(__file__)
        word_tokenize_model.load(join(wd, "models", "ws_crf_vlsp2013_20230727"))
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
