# main function
import os
from os.path import join
import yaml


def generate_prompt_for_new_intents(data):
    # Generate prompt for new intents
    content = ""
    content += "dataset.yaml\n"
    intents = [f"- name: {item['name']}" for item in data]
    content += "```\n" + "\n".join(intents) + "\n```"
    content2 = """
Generate more new 80 topics following this data (except product and smalltalk), give result in yaml format

Output format
```
- name: subject
- name: subject.category
- name: subject.category.subcategory
```

Requirement 1: subject, category and subcategory is lowercase, if it contains more tokens, tokens is concate with _
Requirement 2: each line is not exist in dataset.yaml file
Requirement 3: if subject and subject.category is exist in dataset.yaml, then don't list them to avoid duplicate
"""
    with open(join(prompt_folder, "prompt_generate_intents.txt"), "w") as f:
        f.write(content)

    with open(join(prompt_folder, "prompt_generate_intents_2.txt"), "w") as f:
        f.write(content2)


def extract_need_improve_intents(data):
    config_file = join(script_folder, "config.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    ignored_intents = [item["name"] for item in config["ignores"]]
    need_improved_intents = [
        intent
        for intent in data
        if (len(intent["examples"]) < 3 or len(intent["responses"]) < 1)
        and intent["name"] not in ignored_intents
    ]
    need_improve_intents = need_improved_intents[:5]
    return need_improve_intents


def generate_prompt_for_new_examples_reponses(data):
    need_improve_intents = extract_need_improve_intents(data)
    content = (
        "```\n"
        + "\n".join([f"- name: {intent['name']}" for intent in need_improve_intents])
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
- name: smalltalk.interests
  examples:
  - Sở thích của bạn là gì?
  - Bạn thích gì?
  - Bạn có thích chơi thể thao không?
  responses:
  - Tôi không biết về sở thích cụ thể của bạn. Bạn có thể miêu tả chi tiết hơn để tôi có thể giúp đỡ hơn?
  - Sở thích của bạn là gì? Tôi có thể giúp gì cho bạn?
```
    """
    with open(join(prompt_folder, "prompt_generate_examples_responses.txt"), "w") as f:
        f.write(content)


def generate_prompt_for_new_examples_responses_bloom(data):
    content = ""
    for item in data[:5]:
        content += "===\n"
        content += "Sinh dữ liệu để huấn luyện chatbot\n"
        content += "examples là những câu user nói với bot\n"
        content += "responses là những câu bot nói với user\n\n"
        content += "- name: " + item["name"] + "\n"
        content += "  examples:\n"
        content += "\n".join(["  - " + example for example in item["examples"]])
        content += "\n"
        content += "  responses:\n"
        content += "\n".join(["  - " + response for response in item["responses"]])
        content += "\n\n"
    need_improve_intents = extract_need_improve_intents(data)
    if len(need_improve_intents) > 0:
        new_intent = need_improve_intents[0]
        content += "-name: " + new_intent["name"] + "\n"
        content += "  examples:\n"
        content += "  - "

    with open(
        join(prompt_folder, "prompt_bloom_generate_examples_responses.txt"), "w"
    ) as f:
        f.write(content)


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

    print("# Generate prompt for new intents")
    generate_prompt_for_new_intents(data)
    print("# Generate prompt for new examples and responses")
    generate_prompt_for_new_examples_reponses(data)
    print("# Generate prompt for bloom")
    generate_prompt_for_new_examples_responses_bloom(data)
