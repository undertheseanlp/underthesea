
class _LazyLabels:
    """Lazy-loading labels that behave like a list"""

    def __init__(self, domain=None):
        self._domain = domain
        self._cache = None

    def _load(self):
        if self._cache is None:
            if self._domain == 'bank':
                from underthesea.pipeline.classification import bank
                self._cache = bank.get_labels()
            else:
                from underthesea.pipeline.classification import sonar_core_1
                self._cache = sonar_core_1.get_labels()
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
        list: A list containing the categories of the sentence

    Examples:
        >>> from underthesea import classify
        >>> classify("Thị trường chứng khoán tăng mạnh")
        'kinh_doanh'
        >>> classify.labels
        ['chinh_tri_xa_hoi', 'doi_song', 'khoa_hoc', 'kinh_doanh', ...]
        >>> classify.bank.labels
        ['ACCOUNT', 'CARD', 'DISCOUNT', ...]
    """
    if X == "":
        return None

    if model == 'prompt':
        from underthesea.pipeline.classification import classification_prompt
        args = {
            "domain": domain
        }
        return classification_prompt.classify(X, **args)

    if domain == 'bank':
        from underthesea.pipeline.classification import bank
        return bank.classify(X)

    from underthesea.pipeline.classification import sonar_core_1
    return sonar_core_1.classify(X)


# Attach labels and domain namespaces
classify.labels = _LazyLabels()
classify.bank = _DomainNamespace('bank')
