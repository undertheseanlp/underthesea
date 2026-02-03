import logging
from pathlib import Path

from underthesea.file_utils import UNDERTHESEA_FOLDER, cached_path

FORMAT = "%(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("underthesea")

# Model configurations
MODELS = {
    "general": {
        "url": "https://github.com/undertheseanlp/underthesea/releases/download/resources/sen-classifier-general-1.0.0-20260203.bin",
        "name": "sen-classifier-general-1.0.0-20260203.bin",
    },
    "bank": {
        "url": "https://github.com/undertheseanlp/underthesea/releases/download/resources/sen-classifier-bank-1.0.0-20260203.bin",
        "name": "sen-classifier-bank-1.0.0-20260203.bin",
    },
}

# Cached classifiers
_classifiers = {}


def _get_model_path(domain):
    """Get local path to model, downloading if necessary."""
    config = MODELS[domain]
    cache_dir = Path(UNDERTHESEA_FOLDER) / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / config["name"]

    if not model_path.exists():
        logger.info(f"Downloading {domain} classifier model...")
        cached_path(config["url"], cache_dir=cache_dir)

    return model_path


def _load_classifier(domain="general"):
    """Load classifier for specified domain."""
    if domain not in _classifiers:
        from underthesea_core import TextClassifier
        model_path = _get_model_path(domain)
        _classifiers[domain] = TextClassifier.load(str(model_path))
    return _classifiers[domain]


def _get_labels(domain="general"):
    """Get labels for specified domain."""
    clf = _load_classifier(domain)
    return list(clf.classes)


class _LazyLabels:
    """Lazy-loading labels that behave like a list"""

    def __init__(self, domain="general"):
        self._domain = domain
        self._cache = None

    def _load(self):
        if self._cache is None:
            self._cache = _get_labels(self._domain)
        return self._cache

    def __repr__(self):
        return repr(self._load())

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())

    def __getitem__(self, key):
        return self._load()[key]

    def __contains__(self, item):
        return item in self._load()


class _DomainNamespace:
    """Namespace for domain-specific functions"""

    def __init__(self, domain):
        self._domain = domain
        self._labels = None

    @property
    def labels(self):
        if self._labels is None:
            self._labels = _LazyLabels(self._domain)
        return self._labels


def classify(X, domain=None, model=None):
    """
    Text classification

    Args:
        X (str): The raw sentence
        domain (str, optional): The domain of the text. Defaults to None.
            Options include:
                - None: general domain
                - 'bank': bank domain
        model (str, optional): The classification model. Defaults to None.
            Options include:
                - None: default underthesea classifier
                - 'prompt': OpenAI prompt model

    Returns:
        str or list: Category label (str for general, list for bank domain)

    Examples:
        >>> from underthesea import classify
        >>> classify("Thị trường chứng khoán tăng mạnh")
        'Kinh doanh'
        >>> classify.labels
        ['Chinh tri Xa hoi', 'Doi song', 'Khoa hoc', 'Kinh doanh', ...]
        >>> classify.bank.labels
        ['ACCOUNT', 'CARD', 'DISCOUNT', ...]
    """
    if X == "":
        return None

    if model == 'prompt':
        from underthesea.pipeline.classification import classification_prompt
        args = {"domain": domain}
        return classification_prompt.classify(X, **args)

    if domain == 'bank':
        clf = _load_classifier('bank')
        return [clf.predict(X)]

    clf = _load_classifier('general')
    return clf.predict(X)


# Attach labels and domain namespaces
classify.labels = _LazyLabels("general")
classify.bank = _DomainNamespace("bank")
