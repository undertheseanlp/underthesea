import os
from os.path import join

from underthesea.file_utils import UNDERTHESEA_FOLDER
from underthesea.utils.col_script import UDDataset


def convert_to_sketchengine(ud_dataset, filepath):
    sentences = []
    for s in ud_dataset[:1000]:
        rows_ = []
        for row in s.rows:
            word, tag, _, _ = row
            lemma = word.lower()
            row_content = "\t".join([word, tag, lemma])
            rows_.append(row_content)
        s_content = "<s>\n" + "\n".join(rows_) + "\n</s>"
        sentences.append(s_content)
    content = "\n".join(sentences)
    with open(filepath, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    file = "wiki_00"
    ud_file = join(UNDERTHESEA_FOLDER, "data", "viwiki-20210720", "ud", "AA", file)
    ud_dataset = UDDataset.load(ud_file)

    WIKI_FOLDER = join(UNDERTHESEA_FOLDER, "data", "viwiki-20210720")
    # SE: sketch engine
    SE_FOLDER = join(WIKI_FOLDER, "se", "AA")
    os.makedirs(SE_FOLDER, exist_ok=True)
    se_file = join(SE_FOLDER, file + ".vert")

    convert_to_sketchengine(ud_dataset, se_file)
