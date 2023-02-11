
import yaml


with open("dataset.yaml", "r") as f:
    data = yaml.safe_load(f)

# count how many items in file dataset.yaml
print(f"Total intents: {len(data)}")

# count how many examples in file dataset.yaml
total_examples = sum([len(intent["examples"]) for intent in data])
print(f"Total examples: {total_examples}")

# count how many responses in file dataset.yaml
total_responses = sum([len(intent["responses"]) for intent in data])
print(f"Total responses: {total_responses}")
