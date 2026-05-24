uts_parser = None


def init_parser():
    from underthesea.models.dependency_parser import DependencyParser
    global uts_parser
    if not uts_parser:
        uts_parser = DependencyParser.load('vi-dp-v1a1')
    return uts_parser


def _format_sentence(values):
    return list(zip(values[1], values[6], values[7]))


def dependency_parse(text):
    from underthesea import word_tokenize  # import here to avoid circular import

    parser = init_parser()

    if isinstance(text, (list, tuple)):
        sentences = [word_tokenize(t) for t in text]
        dataset = parser.predict(sentences)
        return [_format_sentence(sentence.values) for sentence in dataset.sentences]

    sentence = word_tokenize(text)
    dataset = parser.predict([sentence])
    return _format_sentence(dataset.sentences[0].values)


# Lazy import visualization functions to avoid circular imports
def __getattr__(name):
    if name in ('render', 'render_tree', 'save', 'display', 'display_tree'):
        from underthesea.pipeline.dependency_parse import visualize
        return getattr(visualize, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
