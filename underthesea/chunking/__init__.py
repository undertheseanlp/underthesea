from underthesea import pos_tag
import sys

if sys.version_info >= (3, 0):
    from .model_crf import CRFChunkingPredictor
else:
    from model_crf import CRFChunkingPredictor


def chunk(sentence, format=None):
    sentence = pos_tag(sentence)
    crf_model = CRFChunkingPredictor.Instance()
    result = crf_model.predict(sentence, format)
    return result



