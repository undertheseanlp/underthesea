import os
from os.path import join, exists

from underthesea.dictionary import Dictionary
from underthesea.file_utils import UNDERTHESEA_FOLDER
from utils import CoreDictionary, DictionaryWord

TS = "202204"
DICTIONARY_PROJECT_FOLDER = join(UNDERTHESEA_FOLDER, "data", f"dictionary-{TS}")
if not exists(DICTIONARY_PROJECT_FOLDER):
    os.mkdir(DICTIONARY_PROJECT_FOLDER)

dictionary = Dictionary.Instance()
core_dictionary = CoreDictionary()

for text in sorted(dictionary.words):
    word = DictionaryWord(text)
    word.senses = dictionary.words[text]
    core_dictionary.add_word(word)
file = join(DICTIONARY_PROJECT_FOLDER, "vlsp_dictionary.yaml")
core_dictionary.to_yaml(file)

file = join(DICTIONARY_PROJECT_FOLDER, "vlsp_dictionary_cp.yaml")
c1 = CoreDictionary.load_yaml(file)
