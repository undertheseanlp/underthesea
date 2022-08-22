from underthesea.utils.vietnamese_ipa import Syllable


def vietnamese_to_ipa(text):
    try:
        syllable = Syllable(text)
        ipa = syllable.generate_ipa()
        return ipa
    except Exception:
        return ""
