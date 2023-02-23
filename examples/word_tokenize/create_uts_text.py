from os.path import join, dirname
from underthesea.file_utils import DATASETS_FOLDER
import random

random.seed(10)
# sampling data
text_file = join(DATASETS_FOLDER, "VNESES", "VNESEScorpus.txt")
with open(text_file) as f:
    lines = f.read().splitlines()
NUM_LONG_TOKENS = 50
NUM_SHORT_TOKENS = 20
long_lines = [
    line
    for line in lines
    if len(line) >= NUM_LONG_TOKENS and line[0].isupper() and line[-1] == "."
]
# get random 1000 lines
random_long_lines = random.sample(long_lines, 5000)
for line in random_long_lines[:20]:
    print(line)


def shortline_conditions(line):
    if len(line) < NUM_SHORT_TOKENS:
        return False
    if len(line) > NUM_LONG_TOKENS:
        return False
    if not line[0].isupper():
        return False
    return True


short_lines = [line for line in lines if shortline_conditions(line)]
random_short_lines = random.sample(short_lines, 5000)
for line in random_short_lines[:20]:
    print(line)

print("Long lines", len(random_long_lines))
print("Short lines", len(random_short_lines))

pwd = dirname(__file__)
tmp = join(pwd, "tmp")
corpus_file = join(tmp, "UTS_Text_v1.txt")

with open(corpus_file, "w") as f:
    lines = random_long_lines + random_short_lines
    content = "\n".join(lines)
    f.write(content)
