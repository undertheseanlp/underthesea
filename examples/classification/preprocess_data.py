from datasets import load_dataset

dataset = load_dataset("uit-nlp/vietnamese_students_feedback")
print(dataset)

print(dataset["train"][0])

sentences = []
for i in range(10):
    item = dataset["train"][i]
    sentence = item["sentence"]
    sentences.append(sentence)
    print(item)

with open("tmp/train.txt", "w") as f:
    content = "\n".join(sentences)
    f.write(content)
