from os.path import dirname, join
from underthesea.models.py_crf_sequence_tagger import PyCRFSequenceTagger

output_dir = join(dirname(__file__), "tmp/ws_20220222")
sentence = "Quỳnh Như tiết lộ với báo Bồ Đào Nha về hành trình làm nên lịch sử"
tokens = sentence.split()

model = PyCRFSequenceTagger()
model.load(output_dir)
y = model.predict(tokens)
for token, x in zip(tokens, y):
    print(token, "\t", x)
