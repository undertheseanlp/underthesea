from os.path import dirname, join

import joblib
from underthesea_core import CRFTagger

from underthesea.pipeline.word_tokenize.regex_tokenize import tokenize

output_dir = join(dirname(__file__), "tmp/ws_202307270300")
sentence = "Quỳnh Như tiết lộ với báo Bồ Đào Nha về hành trình làm nên lịch sử"
sentence = "Thời Trần, những người đứng đầu xã được gọi là Xã quan."
sentence = "Phổ là bang lớn nhất và mạnh nhất trong Liên bang Đức (chiếm 61% dân số và 64% lãnh thổ)."
tokens = tokenize(sentence)
tokens_ = [[token] for token in tokens]

model = CRFTagger()
model.load(join(output_dir, "models.bin"))
features = joblib.load(join(output_dir, "features.bin"))
dictionary = joblib.load(join(output_dir, "dictionary.bin"))
model.set_featurizer(features, dictionary)
y = model.predict(tokens_)
for token, x in zip(tokens, y):
    print(token, "\t", x)
