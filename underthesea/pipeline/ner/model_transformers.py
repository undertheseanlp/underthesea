from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

model_repo = "undertheseanlp/vietnamese-ner-v1.4.0a2"
model_fine_tuned = AutoModelForTokenClassification.from_pretrained(model_repo)
tokenizer = AutoTokenizer.from_pretrained(model_repo)
nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)

if __name__ == '__main__':
    example = "Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump"
    ner_results = nlp(example)
    for entity in ner_results:
        print(entity)
