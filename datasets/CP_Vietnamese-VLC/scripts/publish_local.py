from os import makedirs, listdir
from os.path import join, dirname

from underthesea.file_utils import DATASETS_FOLDER

corpus = join(dirname(dirname(__file__)), "corpus")
corpus_path = join(DATASETS_FOLDER, "CP_Vietnamese-VLC-1.0.0-alpha.1")
makedirs(corpus_path, exist_ok=True)
files = listdir(corpus)
files = [join(corpus, _) for _ in files]

dst = join(corpus_path, "corpus.txt")
with open(dst, "w") as f:
    f.write("")

with open(dst, "a") as outfile:
    for filename in files:
        print(filename)
        with open(filename) as infile:
            text = infile.read()
            outfile.write(text + '\n')
