from os.path import dirname, join

from underthesea.utils.col_dictionary import Dictionary

PROJECT_FOLDER = dirname(dirname(dirname(__file__)))
DATASETS_FOLDER = join(PROJECT_FOLDER, "datasets")
COL_FOLDER = join(DATASETS_FOLDER, "UD_Vietnamese-COL")
DICTIONARY_FOLDER = join(DATASETS_FOLDER, "UD_Vietnamese-COL", "dictionary")
DICTIONARY_FILE = join(DICTIONARY_FOLDER, "202108.yaml")

if __name__ == '__main__':
    dictionary = Dictionary.load(DICTIONARY_FILE)
    dictionary.describe()
    for pos in ["verb", "noun", "adjective", "pronoun", "preposition"]:
        dictionary.import_words(join(DICTIONARY_FOLDER, "data", f"words_{pos}_candidates.xlsx"))
    dictionary.describe()
    dictionary.save(join(DICTIONARY_FOLDER, "202108.yaml"))
