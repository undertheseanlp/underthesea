_translator = None


def translate(text, source_lang='vi', target_lang='en'):
    """
    Translate text between Vietnamese and English.

    Parameters
    ----------
    text : str
        Text to translate
    source_lang : str
        Source language code ('vi' or 'en'). Default: 'vi'
    target_lang : str
        Target language code ('en' or 'vi'). Default: 'en'

    Returns
    -------
    str
        Translated text

    Examples
    --------
    >>> from underthesea import translate
    >>> translate("Xin chào")
    'Hello'
    >>> translate("Hello", source_lang='en', target_lang='vi')
    'Xin chào'
    """
    global _translator
    if _translator is None:
        # Import the deep-learning backend only when translation is requested.
        # Keeping this import out of the module scope prevents a regular
        # ``import underthesea`` from eagerly loading transformers and torch.
        from .envit5 import EnviT5Translator

        _translator = EnviT5Translator()
    return _translator.translate(text, source_lang, target_lang)
