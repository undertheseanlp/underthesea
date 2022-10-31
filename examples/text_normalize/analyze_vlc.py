# Analyze VLC dataset
from os.path import join
from underthesea.file_utils import DATASETS_FOLDER
import pandas as pd

counter = {}
CORPUS_FILE = join(DATASETS_FOLDER, "CP_Vietnamese-VLC-1.0.0-alpha.1", "corpus.txt")
with open(CORPUS_FILE) as f:
    text = f.read()
    tokens = text.lower().split()
    for token in tokens:
        for punct in [",", "\"", ".", ";", ":", "(", ")", "%", "+", "-", "“", "”"]:
            token = token.replace(punct, "")
        if token not in counter:
            counter[token] = 0
        counter[token] += 1

data = {"token": list(counter.keys()), "count": list(counter.values())}
df = pd.DataFrame(data).sort_values(by="count", ascending=False)
df.to_excel("tmp/analyze_vlc.xlsx", index=False)
