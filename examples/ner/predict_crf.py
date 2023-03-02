#@title Predict

from os.path import dirname, join
from underthesea.models.fast_crf_sequence_tagger import FastCRFSequenceTagger
from underthesea import word_tokenize
from underthesea import pos_tag

working_dir = dirname(__file__)
output_dir = join(working_dir, "tmp/ner_20220303")
# sentence = "Quỳnh Như tiết lộ với báo Bồ Đào Nha về hành trình làm nên lịch sử" #@param {type: "string"}
sentence = "Quỳnh Như tiết lộ với báo Bồ Đào Nha về hành trình làm nên lịch sử" #@param {type: "string"}
tokens = pos_tag(sentence)

model = FastCRFSequenceTagger()
model.load(output_dir)
y = model.predict(tokens)
for token, x in zip(tokens, y):
    print(token, "\t", x)