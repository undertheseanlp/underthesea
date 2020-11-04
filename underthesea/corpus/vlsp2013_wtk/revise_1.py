from os import makedirs
from os.path import basename, join, exists
from shutil import copyfile, rmtree

from underthesea.corpus.validate_corpus import validate_corpus_exist
from underthesea.feature_engineering.text import Text
from underthesea.file_utils import DATASETS_FOLDER


def revise_vlsp2013_wtk_dataset(source_file, dest_file):
    copyfile(source_file, dest_file)
    with open(source_file) as f:
        content = Text(f.read())
        sentences = content.split("\n\n")
    if basename(source_file) == "train.txt":
        corpus_id = "train"
    else:
        corpus_id = "test"
    last_index = len(sentences) - 1
    with open(dest_file, "w") as f:
        for i, sentence in enumerate(sentences):
            nodes = sentence.split("\n")
            nodes = [node.split("\t") for node in nodes]
            send_id = f"# sent_id = {corpus_id}-s{i + 1}\n"
            text = " ".join([node[0] for node in nodes])
            text = f"# text = {text}\n"
            content = send_id + text + sentence
            if i != last_index:
                content += "\n\n"
            else:
                content += "\n"
            f.write(content)


def revise_corpus_vlsp2013_wtk():
    source_corpus = "VLSP2013-WTK"
    validate_corpus_exist(source_corpus)
    revise_version = 2
    dest_corpus = f"{source_corpus}-R{revise_version}"
    source_folder = join(DATASETS_FOLDER, source_corpus)
    dest_folder = join(DATASETS_FOLDER, dest_corpus)
    if exists(dest_folder):
        rmtree(dest_folder)
    makedirs(dest_folder)
    files = ["train.txt", "test.txt"]
    for file in files:
        revise_vlsp2013_wtk_dataset(join(source_folder, file), join(dest_folder, file))
