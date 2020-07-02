from os.path import join, dirname
from underthesea.reader.dictionary_loader import DictionaryLoader

words = DictionaryLoader(join(dirname(__file__), "Viet74K.txt")).words
lower_words = set([word.lower() for word in words])


def text_lower(word):
    return word.lower()


def text_isdigit(word):
    return word.isdigit()


def text_isallcap(word):
    for letter in word:
        if not letter.istitle():
            return False
    return True


def text_istitle(word):
    if len(word) == 0:
        return False
    try:
        titles = [s[0] for s in word.split(" ")]
        for token in titles:
            if token[0].istitle() is False:
                return False
        return True
    except Exception:
        return False


def text_is_in_dict(word):
    return str(word.lower() in lower_words)


functions = {
    "lower": text_lower,
    "istitle": text_istitle,
    "isallcap": text_isallcap,
    "isdigit": text_isdigit,
    "is_in_dict": text_is_in_dict
}
