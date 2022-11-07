# -*- coding: utf-8 -*-
from underthesea import chunk
from .model_crf import CRFNERPredictor


def ner(sentence, format=None, deep=False):
    """
    Location and classify named entities in text

    Parameters
    ==========

    sentence: {unicode, str}
        raw sentence

    Returns
    =======
    tokens: list of tuple with word, pos tag, chunking tag, ner tag tagged sentence

    Examples
    --------

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import ner
    >>> sentence = "Ông Putin ca ngợi những thành tựu vĩ đại của Liên Xô"
    >>> ner(sentence)
    [('Ông', 'Nc', 'B-NP', 'O'),
    ('Putin', 'Np', 'B-NP', 'B-PER'),
    ('ca ngợi', 'V', 'B-VP', 'O'),
    ('những', 'L', 'B-NP', 'O'),
    ('thành tựu', 'N', 'B-NP', 'O'),
    ('vĩ đại', 'A', 'B-AP', 'O'),
    ('của', 'E', 'B-PP', 'O'),
    ('Liên Xô', 'Np', 'B-NP', 'B-LOC')]
    """
    if not deep:
        sentence = chunk(sentence)
        crf_model = CRFNERPredictor.Instance()
        result = crf_model.predict(sentence, format)
        return result
    else:
        from .model_transformers import nlp
        output = nlp(sentence)
        if len(output) == 0:
            return []
        entities = [output[0]]
        for item in output[1:]:
            if item["word"].startswith("##"):
                entities[-1]["word"] = entities[-1]["word"] + item["word"][2:]
                entities[-1]["end"] = item["end"]
            else:
                entities.append(item)
        return entities
