# Cross validate with a dependency corpus
from os.path import join, dirname, basename

from termcolor import colored
from underthesea.file_utils import CACHE_ROOT
import joblib

CORPUS_FOLDER = join(CACHE_ROOT, "datasets", "VLSP2020-DP")
train_file = join(CORPUS_FOLDER, "train.txt")
test_file = join(CORPUS_FOLDER, "test.txt")
words = {}
dict_file = join(dirname(dirname(__file__)), "datasets", "DI_Vietnamese-UVD", "UVD.bin")
MAX_SHOW_ERRORS = 100
total_errors = 0

punct = {"!", "/", ",", ".", "...", "?", "-", "\"", "-", ":", "(", ")", "–", "&", ";", "‘", "’", "+"}
specials = {"rbkt", "lbkt"}


def warn(file, line_number, message, type=None):
    global total_errors
    text = ""
    if type:
        text = f"[{type}] "
    text += basename(file) + ":" + str(line_number)
    if total_errors < MAX_SHOW_ERRORS:
        print(colored(text, 'red'), colored(message, 'red'))

    total_errors += 1


def load_dictionary(dict_file):
    dict = joblib.load(dict_file)
    return dict


def load_corpus(file):
    global words
    content = open(file).read().strip()
    sentences = content.split("\n\n")
    for s in sentences:
        nodes = s.split("\n")
        for node in nodes:
            items = node.split("\t")
            word = items[2]
            tag = items[3]
            if word not in words:
                words[word] = set()
            words[word].add(tag)


if __name__ == '__main__':
    load_corpus(train_file)
    load_corpus(test_file)
    dict = load_dictionary(dict_file)
    for word in words:
        if word in specials:
            continue
        if word in punct:
            continue
        tags = list(words[word])
        ignores_tags = {
            "ADJ", "ADV", "INTJ", "NOUN", "PROPN", "PRON", "SYM", "X", "N:G", "VERB:G", "NY",
            "N", "NB", "NNPy",
            "NNP", "NNPy",
            "V", "VERB",
            "Num", "NUMx", "NUM", "NUMX"
        }
        if tags[0] in ignores_tags:
            continue
        if word not in dict:
            warn("", "", f"Dict is not contains {word}")
            print(f"{word} -> {words[word]}")

    if total_errors > 0:
        print(colored(f"\n[x] VALIDATE ERRORS: {total_errors} errors", 'red'))
    else:
        print("\n[+] VALIDATE SUCCESS")
