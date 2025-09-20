import yaml

with open("domain.yml", "r") as f:
    data = yaml.safe_load(f)
    num_intents = len(data['intents'])

with open("data/rules.yml", "r") as f:
    data = yaml.safe_load(f)
    num_rules = len(data['rules'])

with open("data/stories.yml", "r") as f:
    data = yaml.safe_load(f)
    num_stories = len(data['stories'])

print(f"{num_intents} intents, {num_stories} stories, {num_rules} rules")
