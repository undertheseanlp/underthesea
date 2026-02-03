import logging
from pathlib import Path

from underthesea.file_utils import UNDERTHESEA_FOLDER, cached_path

FORMAT = "%(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("underthesea")

classifier = None

MODEL_URL = "https://github.com/undertheseanlp/underthesea/releases/download/resources/sen-classifier-general-1.0.0-20260203.bin"
MODEL_NAME = "sen-classifier-general-1.0.0-20260203.bin"


def _get_model_path():
    """Get local path to model, downloading if necessary."""
    cache_dir = Path(UNDERTHESEA_FOLDER) / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / MODEL_NAME

    if not model_path.exists():
        logger.info("Downloading general classifier model...")
        cached_path(MODEL_URL, cache_dir=cache_dir)

    return model_path


def _load_classifier():
    global classifier
    if classifier is None:
        from underthesea_core import TextClassifier
        model_path = _get_model_path()
        classifier = TextClassifier.load(str(model_path))
    return classifier


def classify(X):
    """Classify text into general categories (news topics).

    Args:
        X: Input text string

    Returns:
        str: Predicted category label
    """
    clf = _load_classifier()
    return clf.predict(X)


def classify_with_confidence(X):
    """Classify text with confidence score.

    Args:
        X: Input text string

    Returns:
        dict: {"category": str, "confidence": float}
    """
    clf = _load_classifier()
    label, score = clf.predict_with_score(X)
    return {"category": label, "confidence": score}


def get_labels():
    """Get all available category labels.

    Returns:
        list: List of category labels
    """
    clf = _load_classifier()
    return list(clf.classes)
