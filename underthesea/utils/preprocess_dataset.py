def preprocess_word_tokenize_dataset_split(data):
    tag_values = data.features['tags'].feature.names
    output = []
    for item in data:
        s = []
        tokens = item['tokens']
        tags = item['tags']
        for token, tag in zip(tokens, tags):
            s.append([token, tag_values[tag]])
        output.append(s)
    return output


def preprocess_word_tokenize_dataset(dataset):
    data = {}
    data["train"] = preprocess_word_tokenize_dataset_split(dataset["train"])
    data["validation"] = preprocess_word_tokenize_dataset_split(dataset["validation"])
    data["test"] = preprocess_word_tokenize_dataset_split(dataset["test"])
    return data
