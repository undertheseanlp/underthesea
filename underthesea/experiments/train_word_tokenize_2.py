from os.path import join

from underthesea.file_utils import CACHE_ROOT
from underthesea.trainers import evaluate_

base_path = "models/wtk_crf"
evaluate_(join(CACHE_ROOT, base_path, "output.txt"))
