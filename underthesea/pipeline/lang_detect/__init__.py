import fasttext
import os
from underthesea.model_fetcher import ModelFetcher

fasttext.FastText.eprint = lambda x: None
lang_detect_model = None


def lang_detect(text):
    global lang_detect_model
    model_name = "LANG_DETECT_FAST_TEXT"
    model_path = ModelFetcher.get_model_path(model_name)
    if not lang_detect_model:
        if not os.path.exists(model_path):
            ModelFetcher.download(model_name)
        try:
            lang_detect_model = fasttext.load_model(str(model_path))
        except Exception:
            pass

    predictions = lang_detect_model.predict(text)
    language = predictions[0][0].replace('__label__', '')
    return language
