from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

model_repo = "rain1024/underthesea_vlsp2016_ner"
model_fine_tuned = AutoModelForTokenClassification.from_pretrained(model_repo)
tokenizer = AutoTokenizer.from_pretrained(model_repo)
nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)

if __name__ == '__main__':
    example = "Bộ Công Thương xóa một tổng cục, giảm nhiều đầu mối"
    ner_results = nlp(example)
    print(ner_results)
