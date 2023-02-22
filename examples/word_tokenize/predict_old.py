from os.path import join, dirname
import pycrfsuite
from py_crf_sequence_tagger import PyCRFSequenceTagger
from underthesea.transformer.tagged import TaggedTransformer


model_path = join(dirname(__file__), "tmp", "model.tmp")
tmp_output_path = join(dirname(__file__), "tmp", "output.txt")

tagger = pycrfsuite.Tagger()
tagger.open(model_path)

features = [
    # word unigram and bigram and trigram
    "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
    "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
    "T[-2,0]", "T[-1,1]", "T[0,2]",

    "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower", "T[2].lower",
    "T[-2,-1].lower", "T[-1,0].lower", "T[0,1].lower", "T[1,2].lower",

    "T[-1].isdigit", "T[0].isdigit", "T[1].isdigit",

    "T[-2].istitle", "T[-1].istitle", "T[0].istitle", "T[1].istitle", "T[2].istitle",
    "T[0,1].istitle", "T[0,2].istitle",

    "T[-2].is_in_dict", "T[-1].is_in_dict", "T[0].is_in_dict", "T[1].is_in_dict", "T[2].is_in_dict",
    "T[-2,-1].is_in_dict", "T[-1,0].is_in_dict", "T[0,1].is_in_dict", "T[1,2].is_in_dict",
    "T[-2,0].is_in_dict", "T[-1,1].is_in_dict", "T[0,2].is_in_dict",
]

tagger = PyCRFSequenceTagger(features)
transformer = TaggedTransformer(features)
sentence = "Quỳnh Như tiết lộ với báo Bồ Đào Nha về hành trình làm nên lịch sử"
tokens = sentence.split()
tokens1 = [[token] for token in tokens]
X = transformer.transform([tokens1])[0]
y = tagger.tag(X)
for token, x in zip(tokens, y):
    print(token, '\t\t', x)
