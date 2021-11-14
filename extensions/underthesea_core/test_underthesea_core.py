from underthesea_core import featurizer

print("-------------------")
print("Test underthesea_core")

print("Call lower function")
sentences = [
    [["Chàng", "X"], ["trai", "X"], ["9X", "X"]],
    [["Khởi", "X"], ["nghiệp", "X"], ["từ", "X"]]
]
features = ["T[0]", "T[1]", "T[2]"]
print(featurizer(sentences, features))
