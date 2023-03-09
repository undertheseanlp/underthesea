import yaml
import os
from os.path import join

script_folder = os.path.dirname(os.path.abspath(__file__))
dataset_folder = join(os.path.dirname(script_folder), "data")
dataset_file = join(dataset_folder, "dataset.yaml")
with open(dataset_file, "r") as f:
    data = yaml.safe_load(f)

target_total_intents = 1000
total_intents = len([item["name"] for item in data])
percent_intents = total_intents / target_total_intents * 100.0
# count how many items in file dataset.yaml
print(f"Total intents\t: {total_intents} ({percent_intents:.2f}%)")

# count how many examples in file dataset.yaml
total_examples = sum([len(intent["examples"]) for intent in data])
target_examples = 3000
percent_examples = total_examples / target_examples * 100.0
print(f"Total examples\t: {total_examples} ({percent_examples:.2f}%)")

# count how many responses in file dataset.yaml
total_responses = sum([len(intent["responses"]) for intent in data])
target_responses = 1500
percent_responses = total_responses / target_responses * 100.0
print(f"Total responses\t: {total_responses} ({percent_responses:.2f}%)")

# generate markdown table for dataset.yaml sorted by intent name
content = ""
content += "| Intent | Length examples |\n"
content += "| ------ | --------------- |\n"
data = sorted(data, key=lambda x: x["name"])
for intent in data:
    content += f"| {intent['name']} | {len(intent['examples'])} |\n"
with open(join(dataset_folder, "dataset.stats.md"), "w") as f:
    f.write(content)
