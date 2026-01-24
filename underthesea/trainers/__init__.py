# Lazy imports to avoid loading dependencies at import time
def __getattr__(name):
    if name == 'ParserTrainer':
        from underthesea.trainers.parser_trainer import ParserTrainer
        return ParserTrainer
    if name == 'DependencyParserTrainer':
        from underthesea.trainers.dependency_parser_trainer import DependencyParserTrainer
        return DependencyParserTrainer
    if name == 'CRFTrainer':
        from underthesea.trainers.crf_trainer import CRFTrainer
        return CRFTrainer
    if name == 'ClassifierTrainer':
        from underthesea.trainers.classifier_trainer import ClassifierTrainer
        return ClassifierTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'CRFTrainer',
    'ClassifierTrainer',
    'ParserTrainer',
    'DependencyParserTrainer'
]
