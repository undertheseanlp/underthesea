from underthesea.utils.vietnamese_ipa import Syllable


def viet2ipa(text: str, *args, **kwargs):
    """Generate ipa of the syllable

    Vietnamese syllabic structure (Anh & Trang 2022)

    syllable = onset + rhyme + tone

    rhyme = medial + nuclear vowel + (coda)

    Args:
        text (str): represents syllable
        dialect (str): Either the `string` `"north"` or `"south"`. Default: `north`
        eight (boolean): If true, use eight tone format, else use six tone format. Default: `False`
        tone (str): Either the `string` `"ipa"` or `"number"`. Default: `number`

    Returns:
        A `string`. Represents ipa of the syllable

    Examples:

        >>> # -*- coding: utf-8 -*-
        >>> from underthesea.pipeline.ipa import viet2ipa
        >>> viet2ipa("trồng")
        tɕoŋ³²
    """
    try:
        syllable = Syllable(text)
        ipa = syllable.generate_ipa(*args, **kwargs)
        return ipa
    except Exception:
        return ""
