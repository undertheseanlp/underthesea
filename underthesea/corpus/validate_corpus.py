import sys
from os import listdir
from os.path import join, basename
from chardet import UniversalDetector

from underthesea.file_utils import DATASETS_FOLDER

SUPPORTED_CORPUS_TYPE = set(["TOKENIZE"])

error_count = 0


def warn(message, level=1, file=None, error_type=None):
    text = ""
    if file:
        text += f"[(in {file})]: "
    text += f"[L{level}"
    if error_type:
        text += f" {error_type}"
    text += "] "
    text += message
    print(text)
    global error_count
    if error_count > 100:
        print("MAX_ERROR_EXCEEDED. Stop")
        sys.exit(1)


def validate_corpus_exist(corpus_name):
    if corpus_name not in listdir(DATASETS_FOLDER):
        print(f"Corpus {corpus_name} is not in existed in {DATASETS_FOLDER}")
        sys.exit(1)


def validate_corpus_type(corpus_type):
    if corpus_type not in SUPPORTED_CORPUS_TYPE:
        print(f"{corpus_type} is not supported")
        sys.exit(1)


# ======================================================================================================================
# LEVEL 1
# TODO: Check NFC Normalize
# ======================================================================================================================
def validate_utf8(file):
    detector = UniversalDetector()
    detector.reset()
    with open(file, "rb") as f:
        for i, line in enumerate(f):
            detector.feed(line)
            if detector.done or i > 10000:
                break
    detector.close()
    result = detector.result
    if not (result["encoding"] == "utf-8" and result["confidence"] >= 0.99):
        warn(message=f"File {file} should encoding with UTF-8", level=1)
        sys.exit(1)


# ======================================================================================================================
# LEVEL 2
# TODO: Validate tokenize
# ======================================================================================================================
def validate_sentence(file):
    global error_count
    base_name = basename(file)
    # sent_id should be valid
    f = open(file)
    sentences = f.read().split("\n\n")
    for sentence in sentences:
        nodes = sentence.split("\n")
        comment_nodes = [node for node in nodes if node.startswith("#")]
        has_sent_id = False
        has_text = False
        for comment_node in comment_nodes:
            if comment_node.startswith("# sent_id = "):
                has_sent_id = True
            if comment_node.startswith("# text = "):
                has_text = True
        if not has_sent_id:
            error_count += 1
            warn(message="Sentence must has sent_id", file=base_name, error_type="Format no-sent_id")
        if not has_text:
            error_count += 1
            warn(message="Sentence must has text", file=base_name,  error_type="Format no-text")
    f.close()


def validate_corpus_content(corpus_name):
    # Level 1
    # Validate NFC Encoding
    corpus_folder = join(DATASETS_FOLDER, corpus_name)
    train = join(corpus_folder, "train.txt")
    test = join(corpus_folder, "test.txt")
    files = [train, test]
    # for file in files:
    #     validate_utf8(file)

    # Level 2
    # Check content
    for file in files:
        validate_sentence(file)
    return True


def validate_corpus(corpus_type, corpus_name):
    global error_count
    error_count = 0
    print(f"Validate {corpus_type} corpus: {corpus_name}")
    validate_corpus_type(corpus_type)
    validate_corpus_content(corpus_name)
