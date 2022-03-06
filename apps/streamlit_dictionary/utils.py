from os.path import abspath, dirname, join

PROJECT_FOLDER = dirname(dirname(abspath(__file__)))
DICTIONARY_FILE = join(PROJECT_FOLDER, "datasets", "UD_Vietnamese-COL", "dictionary", "dictionary.yaml")
