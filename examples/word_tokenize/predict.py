from os.path import dirname, join
from underthesea.models.fast_crf_sequence_tagger import FastCRFSequenceTagger
from underthesea.pipeline.word_tokenize.regex_tokenize import tokenize

output_dir = join(dirname(__file__), "tmp/ws_202307270300")
sentence = "Quỳnh Như tiết lộ với báo Bồ Đào Nha về hành trình làm nên lịch sử"
sentence = "Thời Trần, những người đứng đầu xã được gọi là Xã quan."
sentence = "Phổ là bang lớn nhất và mạnh nhất trong Liên bang Đức (chiếm 61% dân số và 64% lãnh thổ)."
tokens = tokenize(sentence)
tokens_ = [[token] for token in tokens]

model = FastCRFSequenceTagger()
model.load(output_dir)
y = model.predict(tokens_)
for token, x in zip(tokens, y):
    print(token, "\t", x)
