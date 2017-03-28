from underthesea.word_sent.model import CRFModel


def word_sent(sentence):
    crf_model = CRFModel()
    return crf_model.predict(sentence)
