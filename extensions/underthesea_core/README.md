# Underthesea Core (fast & fun)

Underthesea Core contains core NLP functions which is optimized in Rust.

## CRFFeaturizer

```
>>> from underthesea_core import CRFFeaturizer
>>> sentences = [
...     [["Chàng", "X"], ["trai", "X"], ["9X", "X"]],
...     [["trường", "X"], ["học", "X"], ["xanh", "X"]]
... ]
>>> feature_configs = [
...      "T[0]", "T[1]", "T[2]",
...      "T[0,1].is_in_dict"
... ]
>>> dictionary = set(["Chàng", "trai", "trường học"])
>>> crf_featurizer = CRFFeaturizer(feature_configs, dictionary)
>>> crf_featurizer.process(sentences)
[
[['T[0]=Chàng', 'T[1]=trai', 'T[2]=9X', 'T[0,1].is_in_dict=False'],
['T[0]=trai', 'T[1]=9X', 'T[2]=EOS', 'T[0,1].is_in_dict=False'],
['T[0]=9X', 'T[1]=EOS', 'T[2]=EOS', 'T[0,1].is_in_dict=EOS']],
[['T[0]=Khởi', 'T[1]=nghiệp', 'T[2]=từ', 'T[0,1].is_in_dict=False'], ['T[0]=nghiệp', 'T[1]=từ', 'T[2]=EOS', 'T[0,1].is_in_dict=False'], ['T[0]=từ', 'T[1]=EOS', 'T[2]=EOS', 'T[0,1].is_in_dict=EOS']], [['T[0]=trường', 'T[1]=học', 'T[2]=từ', 'T[0,1].is_in_dict=True'], ['T[0]=học', 'T[1]=từ', 'T[2]=EOS', 'T[0,1].is_in_dict=False'], ['T[0]=từ', 'T[1]=EOS', 'T[2]=EOS', 'T[0,1].is_in_dict=EOS']]
]
```