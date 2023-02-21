from os.path import join, dirname
import pycrfsuite
from crf_sequence_tagger import CRFSequenceTagger
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

# tagger = CRFSequenceTagger(features)
transformer = TaggedTransformer(features)
sentence = "Toàn bộ 5/5 tuyến cáp quang biển từ Việt Nam đi quốc tế đều đang gặp sự cố kế hoạch"
tokens = sentence.split()
X = transformer.transform([tokens], contain_labels=False)
print(X)
y = tagger.tag(X[0])
print(y)