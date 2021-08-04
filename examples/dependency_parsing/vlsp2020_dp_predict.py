from os.path import join

from underthesea.file_utils import MODELS_FOLDER
from underthesea.models.dependency_parser import DependencyParser

base_path = join(MODELS_FOLDER, 'parsers', 'vi-dp-v1.3.2a2')
parser = DependencyParser.load(base_path)
sentences = [
    ['Đó', 'là', 'kết quả', 'của', 'cuộc', 'vật lộn', 'bền bỉ', 'gần', '17', 'năm', 'của', 'Huỳnh Đỗi', '.']
]
dataset = parser.predict(sentences)
print(dataset.sentences)
