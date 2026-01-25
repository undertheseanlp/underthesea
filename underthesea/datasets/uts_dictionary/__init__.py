"""UTS Dictionary - Open Vietnamese Dictionary

A comprehensive Vietnamese dictionary with 72,547 words, designed for NLP applications.
Data is hosted on HuggingFace: https://huggingface.co/datasets/undertheseanlp/UTS_Dictionary
"""

_dictionary_instance: "UTSDictionary | None" = None


class UTSDictionary:
    """Vietnamese Dictionary loaded from HuggingFace.

    This dictionary contains 72,547 Vietnamese words and phrases.
    Data is lazily loaded on first access.

    Example:
        >>> from underthesea.datasets.uts_dictionary import UTSDictionary
        >>> dictionary = UTSDictionary()
        >>> words = dictionary.words
        >>> len(words)
        72547
        >>> "xin chào" in dictionary
        True
    """

    REPO_ID = "undertheseanlp/UTS_Dictionary"
    DATA_FILE = "data/data.txt"
    VERSION = "1.0.0"

    def __init__(self):
        self._words: list[str] | None = None
        self._word_set: set | None = None

    def _load(self):
        """Load dictionary from HuggingFace."""
        if self._words is not None:
            return

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "The 'huggingface_hub' package is required for UTSDictionary. "
                "Install it with: pip install huggingface_hub"
            ) from e

        data_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.DATA_FILE,
            repo_type="dataset"
        )

        with open(data_path, encoding="utf-8") as f:
            self._words = [line.strip() for line in f if line.strip()]
        self._word_set = set(self._words)

    @property
    def words(self) -> list[str]:
        """Get list of all words in the dictionary.

        Returns:
            List of Vietnamese words/phrases.
        """
        self._load()
        return self._words

    def __len__(self) -> int:
        """Return number of words in the dictionary."""
        self._load()
        return len(self._words)

    def __contains__(self, word: str) -> bool:
        """Check if a word exists in the dictionary.

        Args:
            word: The word to check.

        Returns:
            True if the word exists, False otherwise.
        """
        self._load()
        return word in self._word_set

    def __iter__(self):
        """Iterate over all words in the dictionary."""
        self._load()
        return iter(self._words)

    def search(self, prefix: str) -> list[str]:
        """Search for words starting with a prefix.

        Args:
            prefix: The prefix to search for.

        Returns:
            List of words starting with the prefix.
        """
        self._load()
        return [word for word in self._words if word.startswith(prefix)]


def get_dictionary() -> UTSDictionary:
    """Get the singleton dictionary instance.

    Returns:
        The UTSDictionary singleton instance.
    """
    global _dictionary_instance
    if _dictionary_instance is None:
        _dictionary_instance = UTSDictionary()
    return _dictionary_instance
