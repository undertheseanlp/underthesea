uts_parser = None


def init_parser():
    from underthesea.models.dependency_parser import DependencyParser
    global uts_parser
    if not uts_parser:
        uts_parser = DependencyParser.load('vi-dp-v1a1')
    return uts_parser


def dependency_parse(text):
    from underthesea import word_tokenize  # import here to avoid circular import

    sentence = word_tokenize(text)
    parser = init_parser()
    dataset = parser.predict([sentence])
    results = dataset.sentences[0].values
    results = list(zip(results[1], results[6], results[7]))
    return results


# Lazy import visualization functions to avoid circular imports
def __getattr__(name):
    if name in ('render', 'render_tree', 'save', 'display', 'display_tree'):
        from underthesea.pipeline.dependency_parse import visualize
        return getattr(visualize, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
