from .restorer import DiacriticsRestorer


def restore_diacritics(text):
    """Restore diacritics for Vietnamese text written without them.

    Uses a lightweight syllable bigram model (dictionary + VNTC corpus)
    with Viterbi decoding. Syllables that already contain diacritics,
    numbers, URLs, emails and punctuation are kept unchanged.

    Args:
        text (str): input text, e.g. text typed without a Vietnamese
            keyboard or scraped from sources where accents are stripped

    Returns:
        str: text with Vietnamese diacritics restored

    Examples:
        >>> from underthesea import restore_diacritics
        >>> restore_diacritics("chung ta co the lam duoc")
        'chúng ta có thể làm được'
    """
    return DiacriticsRestorer.Instance().restore(text)
