from os.path import dirname, join
from underthesea.models.fast_crf_sequence_tagger import FastCRFSequenceTagger

output_dir = join(dirname(__file__), "tmp/ws_20220222")
sentence = "Quỳnh Như tiết lộ với báo Bồ Đào Nha về hành trình làm nên lịch sử"
tokens = sentence.split()
tokens_ = [[token] for token in tokens]

model = FastCRFSequenceTagger()
model.load(output_dir)
y = model.predict(tokens_)
for token, x in zip(tokens, y):
    print(token, "\t", x)
