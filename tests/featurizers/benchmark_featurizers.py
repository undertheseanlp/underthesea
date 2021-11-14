import pstats
from random import choice

from underthesea_core import featurizer
import cProfile
from underthesea.transformer.tagged import TaggedTransformer

templates = [
    "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower", "T[2].lower",

    "T[-1].isdigit", "T[0].isdigit", "T[1].isdigit",

    "T[-1].istitle", "T[0].istitle", "T[1].istitle",
    "T[0,1].istitle", "T[0,2].istitle",

    # word unigram and bigram and trigram
    "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
    "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
    "T[-2,0]", "T[-1,1]", "T[0,2]",
]
transformer = TaggedTransformer(templates)
# 2M tokens
n_sentences = 10000
max_tokens = 200
words = ["Messi", "đạt", "giải", "quả", "bóng", "vàng"]
sentences = []
for i in range(n_sentences):
    word = choice(words)
    sentence = [[word, "X"]] * max_tokens
    sentences.append(sentence)

print('Old')

old_profiler = cProfile.Profile()
old_profiler.enable()
transformer.transform(sentences)
old_profiler.disable()
old_stats = pstats.Stats(old_profiler).sort_stats('tottime')
old_stats.print_stats()

print('New')
new_profiler = cProfile.Profile()
new_profiler.enable()
featurizer(sentences, templates)
new_profiler.disable()
new_stats = pstats.Stats(new_profiler).sort_stats('tottime')
new_stats.print_stats()

old_time = old_stats.total_tt
new_time = new_stats.total_tt
print('Ratio', old_time / new_time, "(", old_time, '->', new_time, ")")
