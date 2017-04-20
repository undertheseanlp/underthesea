import pycrfsuite
from underthesea import word_sent
from underthesea.pos_tag.transformer import Transformer

model = pycrfsuite.Tagger()
model.open("crf-model-1")
template = [
    "T[0]", "T[0].lower", "T[-1].lower", "T[1].lower",
    "T[0].istitle", "T[-1].istitle", "T[1].istitle",
    "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",  # unigram
    "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",  # bigram
    "T[-1][1]", "T[-2][1]", "T[-3][1]",  # dynamic feature
    "T[-3,-2][1]", "T[-2,-1][1]",
    "T[-3,-1][1]"
]


def sentence_to_tupple(sentence):
    return [(token, 'N') for token in sentence.split(' ')]


def predict(sentence, option):
    sentence = word_sent(sentence, True)
    original_sentence = []
    for word in sentence.split(' '):
        if '_' in word:
            original_sentence.append(word.replace('_', ' '))
        else:
            original_sentence.append(word)
    sentence = sentence_to_tupple(sentence)
    sentence = Transformer.extract_features_2(sentence, template)
    y = model.tag(sentence)
    if option:
        output = [(i + 1, original_sentence[i], y[i]) for i in range(len(sentence))]
        return output
    else:
        output = [original_sentence[i] + "/" + y[i] for i in range(len(sentence))]
        return " ".join(output)
