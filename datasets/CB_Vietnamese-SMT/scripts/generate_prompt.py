# main function
import os
from os.path import join
import yaml


if __name__ == "__main__":
    # load dataset
    # read text file dataset.yaml
    script_folder = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = join(os.path.dirname(script_folder), "data")
    prompt_folder = join(os.path.dirname(script_folder), "prompts")

    data_file = join(dataset_folder, "dataset.yaml")
    with open(data_file, "r") as file:
        content = file.read()
        data = yaml.safe_load(content)

    content = "```\n" + content.strip() + "\n```"
    content += "\n\nSinh ra thêm 10 intents mới cho chatbot theo định dạng trên, đưa kết quả dưới dạng yaml"
    with open(join(prompt_folder, "prompt.txt"), "w") as f:
        f.write(content)

    # Generate prompt for new intents
    intents = [f"- name: {item['name']}" for item in data]
    content = "```\n" + "\n".join(intents) + "\n```"
    content += "\n" + "Sinh ra thêm 10 intents mới cho chatbot "
    with open("prompt_generate_intents.txt", "w") as f:
        f.write(content)

    # Template 3: Generate prompt for new examples and responses
    # read config.yaml file
    config_file = join(script_folder, "config.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    ignored_intents = [item["name"] for item in config["ignores"]]
    need_improved_intents = [
        intent
        for intent in data
        if (len(intent["examples"]) < 3 or len(intent["responses"]) < 1) and intent["name"] not in ignored_intents
    ]
    need_improved_intents = need_improved_intents[:5]
    content = (
        "```\n"
        + "\n".join([f"- name: {intent['name']}" for intent in need_improved_intents])
        + "\n```"
    )
    content += "\n\n"
    content += "Sinh ra thêm 5 examples và 2 responses cho mỗi intent theo định dạng"
    content += """
- name: intent_name
  examples:
  - example 1
  - example 2
  responses:
  - response 1
  - response 2
"""
    content += """\nVí dụ:
```
- name: smalltalk.greeting.hello
  examples:
  - xin chào
  - chào
  - chào buổi sáng
  responses:
  - Xin chào! Tôi có thể giúp gì cho bạn hôm nay?
  - Chào! Tôi có thể giúp gì cho bạn?
  - Chào buổi sáng! Cần giúp gì tôi có thể hỗ trợ?
```
    """
    with open(join(prompt_folder, "prompt_generate_examples_responses.txt"), "w") as f:
        f.write(content)
