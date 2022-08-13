# Underthesea Core (fast & fun)

## Usage

CRFFeaturizer

```python
>>> from underthesea_core import CRFFeaturizer
>>> features = ["T[-1]", "T[0]", "T[1]"]
>>> dictionary = set(["sinh viên"])
>>> featurizer = CRFFeaturizer(features, dictionary)
>>> sentences = [[["sinh", "X"], ["viên", "X"], ["đi", "X"], ["học", "X"]]]
>>> featurizer.process(sentences)
[[['T[-1]=BOS', 'T[0]=sinh', 'T[1]=viên'],
  ['T[-1]=sinh', 'T[0]=viên', 'T[1]=đi'],
  ['T[-1]=viên', 'T[0]=đi', 'T[1]=học'],
  ['T[-1]=đi', 'T[0]=học', 'T[1]=EOS']]]
```
