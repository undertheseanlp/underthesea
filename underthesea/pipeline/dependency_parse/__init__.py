from underthesea import word_tokenize
from underthesea.models.dependency_parser import DependencyParser

uts_parser = None


def init_parser():
    global uts_parser
    if not uts_parser:
        uts_parser = DependencyParser.load('vi-dp-v1a1')
    return uts_parser


def dependency_parse(text):
    sentence = word_tokenize(text)
    parser = init_parser()
    dataset = parser.predict([sentence])
    results = dataset.sentences[0].values
    results = list(zip(results[1], results[6], results[7]))
    return results
