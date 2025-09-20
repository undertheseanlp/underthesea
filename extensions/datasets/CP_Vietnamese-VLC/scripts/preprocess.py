from os import listdir
from os.path import abspath, dirname, join
import re

CWD = abspath(dirname(__file__))
RAW_FOLDER = join(dirname(CWD), "output", "raw")
PROCESSED_FOLDER = join(dirname(CWD), "output", "processed")

for file in listdir(RAW_FOLDER):
    pattern = re.compile(r"\s+")
    with open(join(RAW_FOLDER, file), "r") as f:
        sentences = f.read().splitlines()
        sentences = [s.strip() for s in sentences]
        sentences = [s for s in sentences if len(s) > 0]
        sentences = [pattern.sub(" ", s) for s in sentences]
        content = "\n".join(sentences)
    out_filepath = join(PROCESSED_FOLDER, file)
    outfile = open(out_filepath, "w")
    outfile.write(content)
    outfile.close()
