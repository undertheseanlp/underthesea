import shutil
from os import makedirs
from os.path import join, dirname

from underthesea.file_utils import DATASETS_FOLDER

corpus_path = join(DATASETS_FOLDER, "UD_Vietnamese-UUD-1.0.1-alpha")
makedirs(corpus_path, exist_ok=True)
data_file = "all.txt"
src = join(dirname(dirname(__file__)), "corpus", "ud", data_file)
dst = join(corpus_path, data_file)
shutil.copyfile(src, dst)
