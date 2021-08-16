import shutil
from os import makedirs
from os.path import realpath, dirname, join
import yaml
import joblib

FOLDER = dirname(dirname(realpath(__file__)))
content = yaml.safe_load(open(join(FOLDER, "corpus", "data.yaml")))

DICT_FOLDER = join(FOLDER, 'datasets', 'DI_Vietnamese-UVD')
shutil.rmtree(DICT_FOLDER, ignore_errors=True)
makedirs(DICT_FOLDER)
joblib.dump(content, join(DICT_FOLDER, 'UVD.bin'))

DICT_LIGHT_FOLDER = join(FOLDER, 'datasets', 'DI_Vietnamese-UVD_Light')
shutil.rmtree(DICT_LIGHT_FOLDER, ignore_errors=True)
makedirs(DICT_LIGHT_FOLDER)
words = set(content.keys())
joblib.dump(words, join(DICT_LIGHT_FOLDER, 'UVD-light.bin'))

print("+ Build Done.")
