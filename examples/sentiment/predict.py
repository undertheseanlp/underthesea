from transformers import AutoTokenizer, AutoModelWithLMHead
from train_gpt2 import GPT2TextClassification

pretrained_model_name = "imthanhlv/gpt2news"
model = GPT2TextClassification.load_from_checkpoint(
    "/home/anhvu2/projects/underthesea/examples/sentiment/outputs/2022-09-09/10-01-41/lightning_logs/version_0/checkpoints/epoch=0-step=5.ckpt")
text = "Nhân viên thì cũng thân thiện, vui vẻ, mọi thứ đều ổn, tôi thấy giá cả như vậy là tương xứng với dịch vụ của khách sạn."
inputs = model.tokenizer([text], return_tensors='pt')["input_ids"]
labels = model(inputs)
print(labels)