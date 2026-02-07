import logging
from pathlib import Path

from underthesea.file_utils import UNDERTHESEA_FOLDER, cached_path

logger = logging.getLogger("underthesea")

MODEL_URL = "https://github.com/undertheseanlp/underthesea/releases/download/resources/sen-sentiment-bank-1.0.0-20260207.bin"
MODEL_NAME = "sen-sentiment-bank-1.0.0-20260207.bin"

_classifier = None


def _get_model_path():
    cache_dir = Path(UNDERTHESEA_FOLDER) / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / MODEL_NAME
    if not model_path.exists():
        logger.info("Downloading bank sentiment model...")
        cached_path(MODEL_URL, cache_dir=cache_dir)
    return model_path


def _load_classifier():
    global _classifier
    if _classifier is None:
        from underthesea_core import TextClassifier
        model_path = _get_model_path()
        _classifier = TextClassifier.load(str(model_path))
    return _classifier


def sentiment(X) -> list[str]:
    clf = _load_classifier()
    return [clf.predict(X)]


def sentiment_with_confidence(X):
    clf = _load_classifier()
    label, score = clf.predict_with_score(X)
    return {"category": label, "confidence": score}


def get_labels():
    clf = _load_classifier()
    return list(clf.classes)
