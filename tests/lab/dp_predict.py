from os.path import join

from underthesea.models.dependency_parser import DependencyParser
from underthesea.file_utils import MODELS_FOLDER

parser = DependencyParser.load(join(MODELS_FOLDER, 'parsers', 'vi-dp-v1a0'))
dataset = parser.predict(
    [['Đó', 'là', 'kết quả', 'của', 'cuộc', 'vật lộn', 'bền bỉ', 'gần', '17', 'năm', 'của', 'Huỳnh Đỗi', '.']]
)
print(dataset.sentences)
