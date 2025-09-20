from dictionary import Dictionary
import pandas as pd

dictionary = Dictionary()
dictionary.load("dictionary")

sources = []
dict_stats = {}
print("Tổng số từ:", len(dictionary.words), "\n")

for word in dictionary.words:
    sources.append(" ".join(word.source))
    for source in word.source:
        if source not in dict_stats:
            dict_stats[source] = 1
        else:
            dict_stats[source] += 1

s = pd.Series(sources)
print(s.value_counts(), "\n")

WIDTH = 12
for dict, count in dict_stats.items():
    print(f"{dict:{WIDTH}}: {count}")

print("\n", s.describe(), "\n")
