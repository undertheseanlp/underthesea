import os

from underthesea_core import FastText

from underthesea.model_fetcher import ModelFetcher

lang_detect_model = None


def lang_detect(text):
    global lang_detect_model
    model_name = "LANG_DETECT_FAST_TEXT"
    model_path = ModelFetcher.get_model_path(model_name)
    if not lang_detect_model:
        if not os.path.exists(model_path):
            ModelFetcher.download(model_name)
        lang_detect_model = FastText.load(str(model_path))

    predictions = lang_detect_model.predict(text, k=1)
    language = predictions[0][0]
    return language
