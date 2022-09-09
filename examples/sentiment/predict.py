from transformers import AutoTokenizer, AutoModelWithLMHead
from train_gpt2 import GPT2TextClassification

pretrained_model_name = "imthanhlv/gpt2news"
module = GPT2TextClassification.load_from_checkpoint(
    "/home/anhvu2/projects/underthesea/examples/sentiment/outputs/2022-09-07/16-16-19/lightning_logs/version_0/checkpoints/epoch=4-step=2240.ckpt")
text = "khánh sạn mường thanh rất tốt"
inputs = tokenizer([text], return_tensors='pt')["input_ids"]
labels = module(inputs)
