from transformers import pipeline

nlp = pipeline("ner")
print(nlp("VinFast đặt cược gì để đấu với các đối thủ tại Mỹ?"))
