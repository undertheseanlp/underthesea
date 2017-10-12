from underthesea import word_sent

from .model_crf import CRFPOSTagPredictor


def pos_tag(sentence):
    sentence = word_sent(sentence)
    crf_model = CRFPOSTagPredictor.Instance()
    result = crf_model.predict(sentence, format)
    return result

