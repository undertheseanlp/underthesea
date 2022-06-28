import re


class TaggedTransformer:
    def __init__(self, templates=None):
        self.templates = [self._extract_template(template) for template in templates]

    def _extract_template(self, template):
        token_syntax = template
        matched = re.match(
            "T\[(?P<index1>\-?\d+)(\,(?P<index2>\-?\d+))?\](\[(?P<column>.*)\])?(\.(?P<function>.*))?",
            template)
        column = matched.group("column")
        column = int(column) if column else 0
        index1 = int(matched.group("index1"))
        index2 = matched.group("index2")
        index2 = int(index2) if index2 else None
        func = matched.group("function")
        return index1, index2, column, func, token_syntax

    def word2features(self, s):
        features = []
        for i, token in enumerate(s):
            tmp = []
            for template in self.templates:
                debug = True
                index1, index2, column, func, token_syntax = template
                if debug:
                    prefix = "%s=" % token_syntax
                else:
                    prefix = ""
                if i + index1 < 0:
                    result = "%sBOS" % prefix
                    tmp.append(result)
                    continue
                if i + index1 >= len(s):
                    result = "%sEOS" % prefix
                    tmp.append(result)
                    continue
                if index2 is not None:
                    if i + index2 >= len(s):
                        result = "%sEOS" % prefix
                        tmp.append(result)
                        continue
                    tokens = [s[j][column] for j in range(i + index1, i + index2 + 1)]
                    word = " ".join(tokens)
                else:
                    try:
                        word = s[i + index1][column]
                    except:
                        pass
                if func is not None:
                    result = functions[func](word)
                else:
                    result = word
                result = "%s%s" % (prefix, result)
                tmp.append(result)
            features.append(tmp)
        return features

    def transform(self, sentences, contain_labels=False):
        X = [self.word2features(sentence) for sentence in sentences]
        if contain_labels:
            y = [self.sentence2labels(s) for s in sentences]
            return X, y
        return X

    def sentence2labels(self, s):
        return [row[-1] for row in s]


import io


def read(filename):
    with io.open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def write(filename, text):
    with io.open(filename, "w", encoding="utf-8") as f:
        f.write(text)


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
            content = read(self.data_file).strip()
            words = content.split("\n")
            self.words_data = words
        return self.words_data


from os.path import join, dirname

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
    except:
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
