import time
from os.path import join

from underthesea.datasets.vlsp2020_dp import VLSP2020_DP_SAMPLE
from underthesea.file_utils import MODELS_FOLDER
from underthesea.models.dependency_parser import DependencyParser
from underthesea.models.embedings import FieldEmbeddings, CharacterEmbeddings
from underthesea.trainers.dependency_parser_trainer import DependencyParserTrainer

corpus = VLSP2020_DP_SAMPLE()

embeddings = [
    FieldEmbeddings(),
    CharacterEmbeddings()
]
parser = DependencyParser(embeddings=embeddings, init_pre_train=True)
trainer: DependencyParserTrainer = DependencyParserTrainer(parser, corpus)
base_path = join(MODELS_FOLDER, 'parsers', 'vi_dp_sample')
trainer.train(
    base_path=base_path,
    max_epochs=1,
    mu=.9  # optimizer parameters
)

time.sleep(10)

parser = DependencyParser.load(base_path)
sentences = [
    ['Đó', 'là', 'kết quả', 'của', 'cuộc', 'vật lộn', 'bền bỉ', 'gần', '17', 'năm', 'của', 'Huỳnh Đỗi', '.']
]
dataset = parser.predict(sentences)
print(dataset.sentences)
