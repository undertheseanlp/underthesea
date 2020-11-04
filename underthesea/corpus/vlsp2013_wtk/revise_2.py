from os import makedirs
from os.path import basename, join, exists, dirname
from shutil import copyfile, rmtree
import yaml
from underthesea.corpus.validate_corpus import validate_corpus_exist
from underthesea.feature_engineering.text import Text
from underthesea.file_utils import DATASETS_FOLDER

CD = dirname(__file__)
DATA_FILE = join(CD, "revise_2_data.yml")


def load_ignores():
    def extract_ids(content):
        ids = content.split(",")
        ids = set([int(id) for id in ids])
        return ids

    with open(DATA_FILE) as f:
        data = yaml.safe_load(f)
    train_ignore_ids = extract_ids(data["train_ignores"])
    test_ignore_ids = extract_ids(data["test_ignores"])
    result = {
        "train": train_ignore_ids,
        "test": test_ignore_ids
    }
    return result


def revise_vlsp2013_wtk_dataset(source_file, dest_file):
    copyfile(source_file, dest_file)
    with open(source_file) as f:
        content = Text(f.read())
        sentences = content.split("\n\n")
    ignores = load_ignores()
    if basename(source_file) == "train.txt":
        corpus_id = "train"
        ignores_id = ignores["train"]
    else:
        corpus_id = "test"
        ignores_id = ignores["test"]
    last_index = len(sentences) - 1
    with open(dest_file, "w") as f:
        for i, sentence in enumerate(sentences):
            id = i + 1
            if id in ignores_id:
                continue
            nodes = sentence.split("\n")
            nodes = [node.split("\t") for node in nodes]
            send_id = f"# sent_id = {corpus_id}-s{id}\n"
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


if __name__ == '__main__':
    revise_corpus_vlsp2013_wtk()
