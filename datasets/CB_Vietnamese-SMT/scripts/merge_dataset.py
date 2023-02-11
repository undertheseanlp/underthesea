import os
import yaml
from pprint import pprint
import inquirer

from os.path import dirname, join


class Intent:
    def __init__(self, name, examples, responses):
        self.name = name
        self.examples = examples
        self.responses = responses

    def __str__(self):
        return f"Intent: {self.name} ({len(self.examples)} examples, {len(self.responses)} responses))"

    def load_intents(filepath):
        with open(filepath, "r") as file:
            yaml_data = yaml.safe_load(file)
        intents = []
        for item in yaml_data:
            intent = Intent(
                item["name"],
                item["examples"] if "examples" in item else [],
                item["responses"] if "responses" in item else [],
            )
            intents.append(intent)
        return intents

    def to_dict(self):
        data = {}
        if self.name is not None:
            data["name"] = self.name
        if self.examples is not None:
            data["examples"] = list(set(self.examples))
        if self.responses is not None:
            data["responses"] = list(set(self.responses))
        return data

    def merge(self, new_intent):
        merged_intent = self
        new_examples = set(new_intent.examples) - set(self.examples)
        if len(new_examples) > 0:
            print("> Intent:", self.name)
            questions = [
                inquirer.Checkbox(
                    "examples",
                    message="New Examples",
                    choices=list(new_examples),
                ),
            ]
            answers = inquirer.prompt(questions)
            merged_intent.examples.extend(answers["examples"])

        new_responses = set(new_intent.responses) - set(self.responses)
        if len(new_responses) > 0:
            print("> Intent:", self.name)
            questions = [
                inquirer.Checkbox(
                    "responses",
                    message="New Responses",
                    choices=list(new_responses),
                ),
            ]
            answers = inquirer.prompt(questions)
            merged_intent.responses.extend(answers["responses"])
        return merged_intent


def merge_intents(base_intents, new_intents):
    # convert base_intents to dict
    base_intents_dict = {}
    for intent in base_intents:
        base_intents_dict[intent.name] = intent

    # convert new_intents to dict
    new_intents_dict = {}
    for intent in new_intents:
        new_intents_dict[intent.name] = intent

    merged_intents = []
    for name in base_intents_dict:
        if name not in new_intents_dict:
            merged_intents.append(base_intents_dict[name])

    new_intents_num = 0
    for name in new_intents_dict:
        if name not in base_intents_dict:
            print("> New Intent name:", name)
            input_response = input("Are you want to create new intent? (yes/no y/n) ")
            if input_response.lower() in ["yes", "y"]:
                new_intent = Intent(name, [], [])
                merged_intent = new_intent.merge(new_intents_dict[name])
                merged_intents.append(merged_intent)
        else:
            base_intent = base_intents_dict[name]
            merged_intent = base_intent.merge(new_intents_dict[name])
            merged_intents.append(merged_intent)

    print(f"There are {new_intents_num} new possible intents")
    return merged_intents


if __name__ == "__main__":
    # create backedup file for dataset.yaml
    script_folder = os.path.dirname(os.path.realpath(__file__))
    data_folder = join(dirname(script_folder), "data")
    dataset_file = join(data_folder, "dataset.yaml")
    dataset_bk_file = join(data_folder, "dataset.bak.yaml")
    new_dataset_file  = join(data_folder, "new_dataset.yaml")
    with open(dataset_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    with open(dataset_bk_file, "w") as file:
        yaml.dump(yaml_data, file, encoding="utf-8", allow_unicode=True)

    base_intents = Intent.load_intents(dataset_file)
    new_intents = Intent.load_intents(new_dataset_file)

    print("Base Intent")
    for intent in base_intents:
        print(intent)

    print("\nNew Intent")
    for intent in new_intents:
        print(intent)

    print("\nMerged Intent")
    merged_intents = merge_intents(base_intents, new_intents)
    merged_intents_list = []
    for intent in merged_intents:
        print(intent)
        merged_intents_list.append(intent.to_dict())

    # save merged_intents to file
    with open(dataset_file, "w") as file:
        yaml.dump(merged_intents_list, file, encoding="utf-8", allow_unicode=True)
