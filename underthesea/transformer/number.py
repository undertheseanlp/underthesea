import sys

class NumberRemover:
    """
    Remove numbers in documents
    """

    def __init__(self):
        pass

    def _remove(self, document):
        numbers = "0123456789"

        if sys.version_info < (3, 0, 0):
            return document.translate(None, numbers)
        else:
            remover = str.maketrans("", "", numbers)
            return document.translate(remover)

    def transform(self, raw_documents):
        """
        Remove number in each document

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode

        Returns
        -------
        X : iterable
            cleaned documents
        """
        return [self._remove(document) for document in raw_documents]
