uts_parser = None


def init_parser():
    from underthesea.models.dependency_parser import DependencyParser
    global uts_parser
    if not uts_parser:
        uts_parser = DependencyParser.load('vi-dp-v1a1')
    return uts_parser


def dependency_parse(text, batch_size=5000, buckets=8):
    """Dependency-parse a single sentence or a batch of sentences.

    Args:
        text (str or list[str]): a raw sentence, or a list of sentences.
        batch_size (int): number of tokens per torch batch (token-level).
            Passed straight to the model. Default: ``5000``.
        buckets (int): number of length buckets used for batching. Default: ``8``.

    Returns:
        For a ``str`` input: a list of ``(word, head, relation)`` tuples
        (backward compatible).
        For a ``list[str]`` input: a list of such lists, one per sentence,
        in the input order.
    """
    from underthesea import word_tokenize  # import here to avoid circular import

    single = isinstance(text, str)
    texts = [text] if single else list(text)
    if not texts:
        return []

    sentences = [word_tokenize(t) for t in texts]
    parser = init_parser()
    dataset = parser.predict(sentences, batch_size=batch_size, buckets=buckets)

    results = []
    for sentence in dataset.sentences:
        values = sentence.values
        results.append(list(zip(values[1], values[6], values[7])))
    return results[0] if single else results


# Lazy import visualization functions to avoid circular imports
def __getattr__(name):
    if name in ('render', 'render_tree', 'save', 'display', 'display_tree'):
        from underthesea.pipeline.dependency_parse import visualize
        return getattr(visualize, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
