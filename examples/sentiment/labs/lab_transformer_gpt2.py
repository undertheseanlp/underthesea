from transformers import AutoTokenizer

text = "tôi đi học"
text = "Tuy nhiên buffet sáng ở đây không được ngon và chưa đa dạng lắm"
max_token_len = 3
tokenizer = AutoTokenizer.from_pretrained("imthanhlv/gpt2news")
tokenizer.pad_token = tokenizer.eos_token
encoding = tokenizer.encode_plus(
    text=text,
    add_special_tokens=True,
    max_length=50,
    return_token_type_ids=False,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)
print(encoding)
