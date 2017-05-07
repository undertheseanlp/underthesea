import pycrfsuite
from os.path import dirname, join

from underthesea import word_sent
from underthesea.pos_tag.transformer import Transformer

template = [
    "T[0]", "T[0].lower", "T[-1].lower", "T[1].lower",
    "T[0].istitle", "T[-1].istitle", "T[1].istitle",
    "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",  # unigram
    "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",  # bigram
    "T[-1][1]", "T[-2][1]", "T[-3][1]",  # dynamic feature
    "T[-3,-2][1]", "T[-2,-1][1]",
    "T[-3,-1][1]"
]


def predict(sentence, text=False):
    """ make output for raw sentence
    :type text: bool
    :type sentence: raw sentence
    :return list tuple if text = true , sentence pos tagged if text = False
    """
    model = pycrfsuite.Tagger()
    folder = dirname(__file__)
    model.open(join(folder, "crf-model-1"))
    words = word_sent(sentence)
    tokens = [(token, '?') for token in sentence]
    tokens = Transformer.extract_features_2(tokens, template)
    tags = model.tag(tokens)
    output = zip(words, tags)
    if not text:
        output
    return 0
