uts_parser = None


def init_parser():
    from underthesea.models.dependency_parser import DependencyParser
    global uts_parser
    if not uts_parser:
        uts_parser = DependencyParser.load('vi-dp-v1a1')
    return uts_parser


def _format_sentence(values):
    return list(zip(values[1], values[6], values[7]))


def dependency_parse(text, batch_size=5000, buckets=8):
    """Dependency parse one or many sentences.

    Batching happens at the torch level inside the model: sentences are
    grouped into ``buckets`` by length and each forward pass holds about
    ``batch_size`` tokens, so a single ``predict`` call runs many sentences
    through the network at once.

    Args:
        text (str or list[str]): a raw sentence, or a list of sentences.
        batch_size (int): number of tokens per torch batch. Default: 5000.
        buckets (int): number of length buckets used for batching. Default: 8.

    Returns:
        For a single ``str``: a list of ``(word, head, relation)`` tuples.
        For a list of sentences: a list of such lists, in input order.
    """
    from underthesea import word_tokenize  # import here to avoid circular import

    parser = init_parser()

    if isinstance(text, (list, tuple)):
        sentences = [word_tokenize(t) for t in text]
        dataset = parser.predict(sentences, batch_size=batch_size, buckets=buckets)
        return [_format_sentence(sentence.values) for sentence in dataset.sentences]

    sentence = word_tokenize(text)
    dataset = parser.predict([sentence], batch_size=batch_size, buckets=buckets)
    return _format_sentence(dataset.sentences[0].values)


# Lazy import visualization functions to avoid circular imports
def __getattr__(name):
    if name in ('render', 'render_tree', 'save', 'display', 'display_tree'):
        from underthesea.pipeline.dependency_parse import visualize
        return getattr(visualize, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
