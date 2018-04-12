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


from underthesea.corpus import DictionaryLoader
words = DictionaryLoader("Viet74K.txt").words
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


def template2features(sent, i, column, index1, index2, func, token_syntax, debug=True):
    """
    :type token: object
    """
    columns = []
    for j in range(len(sent[0])):
        columns.append([t[j] for t in sent])
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


def word2features(sent, i, template_function):
    features = []
    for token_function in template_function:
        column, index1, index2, func, token_syntax = token_function
        features.extend(template2features(sent, i, column, index1, index2, func, token_syntax))
    return features
