
class _LazyLabels:
    """Lazy-loading labels that behave like a list"""

    def __init__(self, domain='general'):
        self._domain = domain
        self._cache = None

    def _load(self):
        if self._cache is None:
            if self._domain == 'bank':
                from underthesea.pipeline.sentiment.bank import get_labels
                self._cache = get_labels()
            else:
                from underthesea.pipeline.sentiment.general import get_labels
                self._cache = get_labels()
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


def sentiment(X, domain='general'):
    """
    Sentiment Analysis

    Parameters
    ==========

    X: str
        raw sentence
    domain: str
        domain of text (bank or general). Default: `general`

    Returns
    =======
        Text: Text of input sentence
        Labels: Sentiment of sentence

    Examples
    --------

        >>> from underthesea import sentiment
        >>> sentiment("Sản phẩm rất tốt")
        'positive'
        >>> sentiment.labels
        ['positive', 'negative']
        >>> sentiment.bank.labels
        ['ACCOUNT#negative', 'CARD#positive', ...]
    """
    if X == "":
        return None
    if domain == 'general':
        from underthesea.pipeline.sentiment.general import sentiment
        return sentiment(X)
    if domain == 'bank':
        from underthesea.pipeline.sentiment.bank import sentiment
        return sentiment(X)


# Attach labels and domain namespaces
sentiment.labels = _LazyLabels()
sentiment.bank = _DomainNamespace('bank')
