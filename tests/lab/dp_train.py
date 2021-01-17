from os.path import join

from underthesea.datasets.vlsp2020_dp import VLSP2020_DP
from underthesea.file_utils import MODELS_FOLDER
from underthesea.models.dependency_parser import BiaffineDependencyParser
from underthesea.trainers.parser_trainer import ParserTrainer

corpus = VLSP2020_DP()

embeddings = 'char'
embed = False
parser = BiaffineDependencyParser(embeddings, embed=False)
trainer = ParserTrainer(parser, corpus)
trainer.train(
    base_path=join(MODELS_FOLDER, 'parsers, dp'),
    max_epochs=1000,
    mu=0  # optimizer parameters
)
