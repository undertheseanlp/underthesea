import sys
from os import listdir
from os.path import join, basename
from chardet import UniversalDetector

from underthesea.word_tokenize import tokenize
from underthesea.feature_engineering.text import Text
from underthesea.file_utils import DATASETS_FOLDER

SUPPORTED_CORPUS_TYPE = set(["TOKENIZE"])

error_count = 0
DEFAULT_MAX_ERROR = 30
MAX_ERROR = DEFAULT_MAX_ERROR


def warn(message, level=1, file=None, line_number=None, error_type=None, sent_id=None, node_number=None):
    global error_count
    global MAX_ERROR
    error_count += 1
    if error_count >= MAX_ERROR:
        print("MAX_ERROR_EXCEEDED. Stop")
        print(f"*** FAILED *** with {error_count} errors")
        sys.exit(1)

    text = ""
    if file:
        text += f"[(in {file})"
    if line_number:
        text += f" Line {line_number}"
    if sent_id:
        text += f" Sent {sent_id}"
    text += "]: "
    text += f"[L{level}"
    if error_type:
        text += f" {error_type}"
    text += "] "
    text += message
    print(text)


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
    base_name = basename(file)
    detector = UniversalDetector()
    detector.reset()
    with open(file, "rb") as f:
        for i, line in enumerate(f):
            detector.feed(line)
            if detector.done or i > 1000:
                break
    detector.close()
    result = detector.result
    if not (result["encoding"] == "utf-8" and result["confidence"] >= 0.99):
        warn(message=f"File {file} should encoding with UTF-8", level=1)
        sys.exit(1)
    with open(file, "r") as f:
        content = f.read()
    normalized_nfc_content = Text(content)
    if normalized_nfc_content != content:
        warn(message=f"File {base_name} should normalized to NFC",
             error_type="Format nfc-normalized-failed",
             file=base_name, level=1)


# ======================================================================================================================
# LEVEL 2
# ======================================================================================================================
def fetch_sentence(file):
    sentence = ""
    with open(file) as f:
        for i, line in enumerate(f):
            if line == "\n":
                yield i, sentence
                sentence = ""
            else:
                sentence += line


def validate_sentence_format(sentence, i_start, file):
    global error_count
    base_name = basename(file)
    nodes = sentence.strip().split("\n")
    comment_nodes = [node for node in nodes if node.startswith("#")]
    has_sent_id = False
    has_text = False
    for comment_node in comment_nodes:
        if comment_node.startswith("# sent_id = "):
            has_sent_id = True
        if comment_node.startswith("# text = "):
            has_text = True
    if not has_sent_id:
        warn(message="Sentence must has sent_id",
             error_type="Format no-sent_id",
             file=base_name, line_number=i_start, level=2)
    if not has_text:
        warn(message="Sentence must has text",
             error_type="Format no-text",
             file=base_name, line_number=i_start, level=2)


# ======================================================================================================================
# LEVEL 3
# ======================================================================================================================
def validate_token(sentence, i_end, file):
    global error_count
    base_name = basename(file)
    nodes = sentence.strip().split("\n")
    i_start = i_end - len(nodes) + 1
    sent_id = None
    for i, node in enumerate(nodes):
        if node.startswith("#"):
            if node.startswith("# sent_id = "):
                sent_id = node[12:]
            continue
        tokens = node.split("\t")
        line_number = i_start + i
        if len(tokens) != 2:
            warn(message="Each row must has 2 tokens",
                 error_type="Format invalid-num-tokens",
                 file=base_name, line_number=line_number, level=2)
        token = tokens[0]
        tag = tokens[1]
        if token.startswith(" ") or token.endswith(" "):
            warn(message=f"token should not start and end with spaces: {token}",
                 error_type="Format token-error",
                 file=base_name, line_number=line_number, level=3)

        if tag not in ["B-W", "I-W"]:
            warn(message=f"Tag must be B-W or I-W, found {tag}",
                 error_type="Format tag-error",
                 file=base_name, line_number=line_number, level=3, sent_id=sent_id)

    content_nodes = [node for node in nodes if not node.startswith("#")]
    content_tokens = [node.split("\t")[0] for node in content_nodes]
    text = " ".join(content_tokens)
    tokenized_tokens = tokenize(text)
    if tokenized_tokens != content_tokens:
        if len(text) > 53:
            message = "tokenized error for text: " + text[:50] + "..."
        else:
            message = "tokenized error for text: " + text
        message += '\n'
        message += '  Actual:' + str(content_tokens)
        message += '\n'
        message += 'Expected:' + str(tokenized_tokens)
        warn(message=message,
             error_type="Format tokenize-error",
             file=base_name, line_number=i_start, level=3, sent_id=sent_id)


def validate_content(file):
    global error_count
    for i, sentence in fetch_sentence(file):
        validate_sentence_format(sentence, i, file)
        validate_token(sentence, i, file)


def validate_corpus_content(corpus_name):
    # Level 1
    # Validate NFC Encoding
    corpus_folder = join(DATASETS_FOLDER, corpus_name)
    train = join(corpus_folder, "train.txt")
    test = join(corpus_folder, "test.txt")
    files = [train, test]
    for file in files:
        validate_utf8(file)

    # Level 2, 3
    # Check content
    for file in files:
        validate_content(file)


def validate_corpus(corpus_type, corpus_name, max_error=DEFAULT_MAX_ERROR):
    global error_count
    global MAX_ERROR
    MAX_ERROR = max_error
    error_count = 0
    print(f"Validate {corpus_type} corpus: {corpus_name}")
    validate_corpus_type(corpus_type)
    validate_corpus_content(corpus_name)

    if error_count == 0:
        print("*** PASSED ***")
    else:
        print(f"*** FAILED *** with {error_count} errors")
