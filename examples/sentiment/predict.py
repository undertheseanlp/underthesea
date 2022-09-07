from transformers import AutoTokenizer, AutoModelWithLMHead
from train_gpt2 import GPT2TextClassification

tokenizer = AutoTokenizer.from_pretrained("imthanhlv/gpt2news")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

gpt2 = AutoModelWithLMHead.from_pretrained("imthanhlv/gpt2news")
gpt2.resize_token_embeddings(len(tokenizer))
gpt2.config.pad_token_id = gpt2.config.eos_token_id
module = GPT2TextClassification.load_from_checkpoint("/home/anhvu2/projects/underthesea/examples/sentiment/outputs/2022-09-07/14-43-11/lightning_logs/version_0/checkpoints/epoch=4-step=3590.ckpt", gpt2=gpt2, num_labels=34)
text = "khánh sạn mường thanh rất tốt"
inputs = tokenizer([text], return_tensors='pt')["input_ids"]
labels = module(inputs)
print(0)