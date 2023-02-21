# ===========================
# token syntax
# ===========================
#         _ row 1
#        /  _ row 2
#       /  /  _ column
#      /  /  /
#    T[0,2][0]
#          .is_digit
#            \_ function
#
# ===========================
# sample tagged sentence
# ===========================
# this     A
# is       B
# a        C
# sample   D
# sentence E


import re

from os.path import join, dirname

from os.path import join, dirname


class DictionaryLoader:
    def __init__(self, filepath):
        """load words from Ho Ngoc Duc's dictionary
        :param str filepath: filename of dictionary data
        :type filepath: str
        """
        data_folder = join(dirname(dirname(__file__)), "data")
        data_file = join(data_folder, filepath)
        self.data_file = data_file
        self.words_data = None

    @property
    def words(self):
        if not self.words_data:
            with open(self.data_file) as f:
                content = f.read().strip()
            words = content.split("\n")
            self.words_data = words
        return self.words_data


words = DictionaryLoader(join(dirname(__file__), "Viet74K.txt")).words
lower_words = set([word.lower() for word in words])


def text_lower(word):
    return word.lower()


def text_isdigit(word):
    return str(word.isdigit())


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
    except:
        return False


def text_is_in_dict(word):
    return str(word.lower() in lower_words)


def apply_function(name, word):
    functions = {
        "lower": text_lower,
        "istitle": text_istitle,
        "isallcap": text_isallcap,
        "isdigit": text_isdigit,
        "is_in_dict": text_is_in_dict
    }
    return functions[name](word)


def template2features(sent, i, token_syntax, debug=True):
    """
    :type token: object
    """
    columns = []
    for j in range(len(sent[0])):
        columns.append([t[j] for t in sent])
    matched = re.match(
        "T\[(?P<index1>\-?\d+)(\,(?P<index2>\-?\d+))?\](\[(?P<column>.*)\])?(\.(?P<function>.*))?",
        token_syntax)
    column = matched.group("column")
    column = int(column) if column else 0
    index1 = int(matched.group("index1"))
    index2 = matched.group("index2")
    index2 = int(index2) if index2 else None
    func = matched.group("function")
    if debug:
        prefix = "%s=" % token_syntax
    else:
        prefix = ""
    if i + index1 < 0:
        return ["%sBOS" % prefix]
    if i + index1 >= len(sent):
        return ["%sEOS" % prefix]
    if index2 is not None:
        if i + index2 >= len(sent):
            return ["%sEOS" % prefix]
        word = " ".join(columns[column][i + index1: i + index2 + 1])
    else:
        word = sent[i + index1][column]
    if func is not None:
        result = apply_function(func, word)
    else:
        result = word
    return ["%s%s" % (prefix, result)]


def word2features(sent, i, template):
    features = []
    for token_syntax in template:
        features.extend(template2features(sent, i, token_syntax))
    return features