from underthesea_core import CRFFeaturizer
print("-------------------")
print("Test underthesea_core")


sentences = [
    [["Chàng", "X"], ["trai", "X"], ["9X", "X"]],
    [["Khởi", "X"], ["nghiệp", "X"], ["từ", "X"]],
    [["trường", "X"], ["học", "X"], ["từ", "X"]]
]
feature_configs = [
    "T[0]", "T[1]", "T[2]",
    "T[0,1].is_in_dict"
]
dictionary = set(["Chàng", "trai", "trường học"])

# print("Call featurizer function")
# print(featurizer(sentences, feature_configs, dictionary))

print("Call CRFFeaturizer function")
crf_featurizer = CRFFeaturizer(feature_configs, dictionary)
print(crf_featurizer.process(sentences))
