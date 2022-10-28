from underthesea.utils.vietnamese_ipa import Syllable


def vietnamese_to_ipa(text):
    """Generate ipa of the syllable
    Vietnamese syllabic structure (Trang 2022)
    syllable = onset + rhyme + tone
    rhyme = medial + nuclear vowel + (coda)

    Args:
        dialect (str): Either the `string` `"north"` or `"south"`
        eight (boolean): If true, use eight tone format, else use six tone format
        tone (str): Either the `string` `"ipa"` or `"number"`

    Returns:
        A `string`. Represents ipa of the syllable
    """
    try:
        syllable = Syllable(text)
        ipa = syllable.generate_ipa()
        return ipa
    except Exception:
        return ""
