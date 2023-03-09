import yaml

if __name__ == "__main__":
    filepath = "dataset.yaml"
    with open(filepath, 'r') as file:
        yaml_data = yaml.safe_load(file)
    print(yaml_data)
