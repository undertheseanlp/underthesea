import yaml
from os.path import join, dirname

# load yaml from string
s = """\
- examples:
  - chào
  - chào buổi sáng
  - xin chào
  name: smalltalk.greeting.hello
  responses:
  - Xin chào! Tôi có thể giúp gì cho bạn hôm nay?
  - Chào buổi sáng! Cần giúp gì tôi có thể hỗ trợ?
  - Chào! Tôi có thể giúp gì cho bạn?
- examples:
  - chào ngày mới
  - buổi sáng vui vẻ
  - ngày mới tốt lành
  name: smalltalk.greeting.goodmorning
  responses:
  - Chào ngày mới! Chúc một ngày tốt lành.
  - Ngày mới tốt lành! Chúc một buổi sáng vui vẻ.
  - Buổi sáng vui vẻ! Chào ngày mới.
"""
data = yaml.load(s, Loader=yaml.FullLoader)
print(data)

wd = dirname(__file__)
rasa_template_folder = join(dirname(wd), "chatbot_template")
nlu_file = join(rasa_template_folder, "data", "nlu.yml")

with open(nlu_file, "w") as f:
    content = "version: \"3.1\"" + "\n"
    
    data = [
            {
                "name": "greet",
                "examples": [
                    "a",
                    "b",
                    "c"
                ]
            },
            {
                "name": "goodbye",
                "examples": [
                    "d",
                    "e",
                    "f"
                ]
            }
    ]

    content += "\nnlu:\n"
    for intent in data:
        content += f"- intent: {intent['name']}\n"
        content += '  examples: |\n'
        content += '\n'.join([f'    - {example}' for example in intent['examples']])
        content += '\n\n'

    f.write(content)

