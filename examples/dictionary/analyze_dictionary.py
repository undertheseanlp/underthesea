import os
from os.path import join, exists

from underthesea.dictionary import Dictionary
from underthesea.file_utils import UNDERTHESEA_FOLDER

TS = "202204"
DICTIONARY_PROJECT_FOLDER = join(UNDERTHESEA_FOLDER, "data", f"dictionary-{TS}")
if not exists(DICTIONARY_PROJECT_FOLDER):
    os.mkdir(DICTIONARY_PROJECT_FOLDER)

dictionary = Dictionary.Instance()
with open(join(DICTIONARY_PROJECT_FOLDER, "dictionary_v1.3.4.txt"), "w") as f:
    for word in sorted(dictionary.words):
        print(word)
        f.write(word + "\n")
