from os.path import join
from underthesea import word_tokenize
from underthesea.file_utils import DATASETS_FOLDER


filepath = join(DATASETS_FOLDER, "LTA", "VNESEScorpus.txt")
count = 0

with open(filepath, "r") as f:
    for line in f:
        count += 1
        if count > 1000000:
            break
        text = line.strip()
        tokens = word_tokenize(text)
        for token in tokens:
            contains_multi_word = False
            multi_word = ""
            if len(token.split(" ")) > 2:
                contains_multi_word = True
                multi_word = token
                break
        if contains_multi_word:
            print(multi_word)
